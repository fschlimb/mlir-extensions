//===- DistCoalesce.cpp - NDArrayToDist Transform  -----*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements transforms of the Dist dialect.
///
/// This pass tries to minimize the number of mesh::ShardOps.
/// Instead of creating a new copy for each repartition, it tries to combine
/// multiple RePartitionOps into one. For this, it computes the local bounding
/// box of several uses of repartitioned copies of the same base araay. It
/// replaces all matched RepartitionOps with one which provides the computed
/// bounding box. Uses of the eliminated RePartitionOps get updated with th
/// appropriate target part as originally used. Right now supported uses are
/// SubviewOps and InsertSliceOps.
///
/// InsertSliceOps are special because they mutate data. Hence they serve as
/// barriers across which no combination of RePartitionOps will happen.
///
/// Additionally, while most other ops do not request a special target part,
/// InsertSliceOps request a target part on the incoming array. This target
/// part gets back-propagated as far as possible, most importantly including
/// EWBinOps.
///
/// Also, as part of this back-propagation, RePartitionOps between two EWBinOps,
/// e.g. those which come from one EWBinOp and have only one use and that in a
/// another EWBinOp get simply erased.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/NDArray/IR/NDArrayOps.h>
#include <imex/Dialect/NDArray/Utils/Utils.h>
#include <imex/Utils/ArithUtils.h>
#include <imex/Utils/PassUtils.h>

#include <mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Mesh/IR/MeshDialect.h>
#include <mlir/Dialect/Mesh/IR/MeshOps.h>
#include <mlir/Dialect/Utils/IndexingUtils.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ShapedOpInterfaces.h>

#include "PassDetail.h"

#include <iostream>
#include <set>
#include <unordered_map>

namespace imex {
namespace dist {

namespace {

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Lowering dist dialect by no-ops
struct DistCoalescePass : public ::imex::DistCoalesceBase<DistCoalescePass> {

  DistCoalescePass() = default;

#if 0
  // returns true if a Value is defined by any of the given operation types
  template <typename T, typename... Ts>
  static ::mlir::Operation *isDefByAnyOf(const ::mlir::Value &val) {
    if (auto res = val.getDefiningOp<T>())
      return res;
    if constexpr (sizeof...(Ts))
      return isDefByAnyOf<Ts...>(val);
    else if constexpr (!sizeof...(Ts))
      return nullptr;
  }

  // returns true if an operation is of any of the given types
  template <typename T, typename... Ts>
  static bool isAnyOf(const ::mlir::Operation *op) {
    if (::mlir::dyn_cast<T>(op))
      return true;
    if constexpr (sizeof...(Ts))
      return isAnyOf<Ts...>(op);
    else if constexpr (!sizeof...(Ts))
      return false;
  }

  static bool isCreator(::mlir::Operation *op) {
    return op &&
           isAnyOf<::imex::ndarray::LinSpaceOp, ::imex::ndarray::CreateOp>(op);
  }

  /// Follow def-chain of given Value until hitting a creation function
  /// or array-returning EWBinOp or EWUnyOp et al
  /// @return defining op
  ::mlir::Operation *getBaseArray(const ::mlir::Value &val) {
    auto defOp = val.getDefiningOp();
    if (isAnyOf<::imex::ndarray::EWBinOp, ::imex::ndarray::EWUnyOp,
                ::imex::ndarray::ReshapeOp, ::imex::ndarray::CopyOp,
                ::imex::ndarray::LinSpaceOp, ::imex::ndarray::CreateOp>(
            defOp)) {
      return defOp;
    } else if (auto op = ::mlir::dyn_cast<::imex::ndarray::SubviewOp>(defOp)) {
      return getBaseArray(op.getSource());
    } else if (auto op =
                   ::mlir::dyn_cast<::imex::ndarray::InsertSliceOp>(defOp)) {
      return getBaseArray(op.getDestination());
    } else if (auto op = ::mlir::dyn_cast<::mlir::mesh::ShardOp>(defOp)) {
      return getBaseArray(op.getSrc());
    } else if (auto op = ::mlir::dyn_cast<::mlir::UnrealizedConversionCastOp>(
                   defOp)) {
      if (op.getInputs().size() == 1) {
        return getBaseArray(op.getInputs().front());
      }
      return defOp;
    } else {
      std::cerr << "oops. Unexpected op found: ";
      const_cast<::mlir::Value &>(val).dump();
      assert(false);
    }
  }

  /// return true if given op comes from a EWOp and has another EWOp
  /// as its single user.
  bool is_temp(::mlir::mesh::ShardOp &op) {
    if (!op->hasAttr("target") && op->hasOneUse() &&
        ::mlir::isa<::imex::dist::EWBinOp, ::imex::dist::EWUnyOp>(
            *op->user_begin()) &&
        ::mlir::isa<::imex::dist::EWBinOp, ::imex::dist::EWUnyOp>(
            op.getSrc().getDefiningOp())) {
      return true;
    }
    return false;
  }

  /// update a SubviewOp with a target part
  /// create and return a new op if the SubviewOp has more than one use.
  ::mlir::Operation *updateTargetPart(::mlir::IRRewriter &builder,
                                      ::imex::dist::SubviewOp op,
                                      const ::mlir::ValueRange &tOffs,
                                      const ::mlir::ValueRange &tSizes) {

    // check if an existing target is the same as ours
    auto offs = op.getTargetOffsets();
    auto szs = op.getTargetSizes();
    if (offs.size() > 0) {
      assert(offs.size() == szs.size());
      ::mlir::SmallVector<::mlir::Operation *> toBeMoved;
      for (size_t i = 0; i < offs.size(); ++i) {
        if ((tOffs[i] != offs[i] || tSizes[i] != szs[i]) && !op->hasOneUse()) {
          // existing but different target -> need a new repartition for our
          // back-propagation
          auto val = op.getSource();
          builder.setInsertionPointAfter(op);

          auto tmp = tOffs[0].getDefiningOp();
          auto &dom = this->getAnalysis<::mlir::DominanceInfo>();
          if (!dom.dominates(tmp, op)) {
            toBeMoved.resize(0);
            if (canMoveAfter(dom, tmp, op, toBeMoved)) {
              ::mlir::Operation *curr = op;
              for (auto dop : toBeMoved) {
                dop->moveAfter(curr);
                curr = dop;
              }
              builder.setInsertionPointAfter(curr);
            } else {
              assert(false && "Not implemented");
            }
          }
          assert(tOffs.size() == tSizes.size());
          auto dynPtType = cloneWithDynEnv(
              mlir::cast<::imex::ndarray::NDArrayType>(val.getType()));
          return builder.create<::imex::mesh::ShardOp>(
              op->getLoc(), dynPtType, val, tOffs, tSizes);
        }
      }
      // if same existing target -> nothing to be done
    } else {
      const int32_t rank = static_cast<int32_t>(tOffs.size());
      const int32_t svRank = op.getStaticSizes().size();
      const bool hasUnitSize =
          mlir::cast<::imex::ndarray::NDArrayType>(op.getResult().getType())
              .hasUnitSize();

      if (svRank == rank || hasUnitSize) {
        if (hasUnitSize) {
          // Here the subview can have a different rank than the target.
          // The target can be empty (all dims have size zero) for example when
          // the source insert_slice is unit-sized and happens on a different
          // prank. In such cases we need to have all zeros in our target (of
          // rank svRank). Otherwise the target size is 1.
          mlir::OpBuilder::InsertionGuard guard(builder);
          if (rank) {
            builder.setInsertionPointAfter(tSizes[0].getDefiningOp());
          } else {
            builder.setInsertionPoint(op);
          }

          // first compute total size of target
          auto loc = op->getLoc();
          auto zero = easyIdx(loc, builder, 0);
          auto one = easyIdx(loc, builder, 1);
          auto sz = one;
          for (auto r = 0; r < rank; ++r) {
            sz = sz * easyIdx(loc, builder, tSizes[r]);
          }
          // check if the target has total size 0
          sz = sz.eq(zero).select(zero, one);
          op->insertOperands(op->getNumOperands(),
                             ::imex::ValVec(svRank, zero.get()));
          op->insertOperands(op->getNumOperands(),
                             ::imex::ValVec(svRank, sz.get()));
        } else {
          // no existing target -> use ours
          op->insertOperands(op->getNumOperands(), tOffs);
          op->insertOperands(op->getNumOperands(), tSizes);
        }

        const auto sSzsName = op.getOperandSegmentSizesAttrName();
        const auto oa = op->getAttrOfType<::mlir::DenseI32ArrayAttr>(sSzsName);
        ::std::array<int32_t, 6> sSzs{oa[0], oa[1],  oa[2],
                                      oa[3], svRank, svRank};
        op->setAttr(sSzsName, builder.getDenseI32ArrayAttr(sSzs));
      } else {
        assert(false && "found dependent operation with different rank, needs "
                        "broadcasting support?");
      }
    }
    return nullptr;
  }

  /// clone subviewops which are returned and mark them "final"
  /// Needed to protect them from being "redirected" to a reparitioned copy
  void backPropagateReturn(::mlir::IRRewriter &builder,
                           ::mlir::func::ReturnOp retOp) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(retOp);
    bool altered = false;
    ::imex::ValVec oprnds;
    ::mlir::SmallVector<::mlir::Operation *> toErase;
    for (auto val : retOp->getOperands()) {
      if (isDist(val)) {
        bool oneUse = true;
        // "skip" casts and observe if this is a single-use chain
        auto castOp = val.getDefiningOp<::mlir::UnrealizedConversionCastOp>();
        while (castOp && castOp.getInputs().size() == 1) {
          if (!castOp->hasOneUse()) {
            oneUse = false;
          }
          val = castOp.getInputs().front();
          castOp = val.getDefiningOp<::mlir::UnrealizedConversionCastOp>();
        }

        if (auto typedOp = val.getDefiningOp<::imex::dist::SubviewOp>()) {
          auto iOp = builder.clone(*typedOp);
          iOp->setAttr("final", builder.getUnitAttr());
          if (oneUse && typedOp->hasOneUse()) {
            toErase.emplace_back(typedOp);
          }
          oprnds.emplace_back(iOp->getResult(0));
          altered = true;
          continue;
        }
      }
      oprnds.emplace_back(val);
    }
    if (altered) {
      retOp->setOperands(oprnds);
      for (auto op : toErase) {
        op->erase();
      }
    }
  }

  /// entry point for back propagation of target shardings.
  void backPropagateSharding(::mlir::IRRewriter &builder,
                             ::mlir::mesh::ShardOp op) {
    ::mlir::Operation *nOp = nullptr;
    auto sharding = op.getSharding();
    if (!sharding) {
      return;
    }
    auto defOp = op.getSrc().getDefiningOp();
    if (defOp) {
      backPropagateSharding(builder, defOp, sharding, nOp);
      assert(nOp == nullptr);
    }
    return;
  }

  /// The actual back propagation of target parts
  /// if meeting a supported op, recursively gets defining ops and back
  /// propagates as it follows only supported ops, all other ops act as
  /// propagation barriers (e.g. InsertSliceOps) on the way it updates target
  /// info on SubviewOps and marks shardOps for elimination
  void backPropagateSharding(::mlir::IRRewriter &builder, ::mlir::Operation *op,
                             ::mlir::Value sharding, ::mlir::Operation *&nOp) {
    nOp = nullptr;
    if (auto typedOp = ::mlir::dyn_cast<::mlir::mesh::ShardOp>(op)) {
      typedOp.getShardingMutable().assign(sharding);
      op = typedOp.getSrc().getDefiningOp();
      assert(op);
    }
    if (auto typedOp = ::mlir::dyn_cast<::imex::ndarray::EWBinOp>(op)) {
      op = typedOp.getLhs().getDefiningOp();
      if (op) {
        backPropagateSharding(builder, op, sharding, nOp);
      }
      op = typedOp.getRhs().getDefiningOp();
    } else if (auto typedOp = ::mlir::dyn_cast<::imex::ndarray::EWUnyOp>(op)) {
      op = typedOp.getSrc().getDefiningOp();
    } else if (auto typedOp =
                   ::mlir::dyn_cast<::mlir::UnrealizedConversionCastOp>(op)) {
      if (typedOp.getInputs().size() == 1) {
        op = typedOp.getInputs().front().getDefiningOp();
      } else {
        op = nullptr;
      }
    } else if (!::mlir::isa<::mlir::mesh::ShardOp>(op)) {
      op = nullptr;
    }
    if (op) {
      backPropagateSharding(builder, op, sharding, nOp);
    }
    return;
  }

  // return ShardOp that annotates the result of a given op
  ::mlir::mesh::ShardOp getShardOp(::mlir::Operation *op) {
    assert(op->hasOneUse() || op->use_empty());
    op = *op->user_begin();
    if (::mlir::isa<::mlir::UnrealizedConversionCastOp>(op)) {
      assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
      assert(op->hasOneUse());
      op = *op->user_begin();
    }
    return ::mlir::dyn_cast<::mlir::mesh::ShardOp>(op);
  }

  // return ShardOp that annotates the given operand/value
  ::mlir::mesh::ShardOp getShardOpOfOperand(::mlir::Value val) {
    auto op = val.getDefiningOp();
    // FIXME as long as we have NDArrays we might meet casts
    if (::mlir::isa<::mlir::UnrealizedConversionCastOp>(op)) {
      assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
      assert(op->hasOneUse() && op->getNumOperands() == 1);
      op = op->getOperand(0).getDefiningOp();
    }
    assert(op->hasOneUse());
    return ::mlir::dyn_cast<::mlir::mesh::ShardOp>(op);
  }

  /// compute target part for a given InsertSliceOp
  ::imex::dist::TargetOfSliceOp computeTarget(::mlir::IRRewriter &builder,
                                              ::imex::ndarray::InsertSliceOp op,
                                              ::mlir::Value sharding) {
    auto shardingOp =
        ::mlir::cast<::mlir::mesh::ShardingOp>(sharding.getDefiningOp());
    auto sOffs = op.getStaticOffsets();
    auto sSizes = op.getStaticSizes();
    auto sStrides = op.getStaticStrides();
    assert(!(::mlir::ShapedType::isDynamicShape(sSizes) ||
             ::mlir::ShapedType::isDynamicShape(sOffs) ||
             ::mlir::ShapedType::isDynamicShape(sStrides)) ||
           (false && "SubviewOp must have dynamic offsets, sizes and strides"));

    auto src = getShardOpOfOperand(op.getDestination()).getSrc();
    return builder.create<::imex::dist::TargetOfSliceOp>(
        op->getLoc(), src, sOffs, sSizes, sStrides, shardingOp.getMeshAttr(),
        shardingOp.getSplitAxes());
  }
#endif // 0

  // This pass tries to combine multiple ShardOps into one.
  // It does not actually erase any ops, but rather annotates some so that
  // later passes will not create actual resharding/communicating ops.
  //
  // Dependent operations (like SubviewOp) get adequately annotated.
  //
  // The basic idea is to compute a the bounding box of several SubViews
  // and use it for a combined ShardOp. Dependent SubviewOps can then
  // extract the appropriate part from that bounding box without further
  // communication/repartitioning.
  //
  // Right now we only support subviews with static indices (offs, sizes,
  // strides).
  //
  // 1. back-propagation of explicit target-parts
  // 2. group SubviewOps
  // 3. create base ShardOp and update dependent SubviewOps
  void runOnOperation() override {
#if 0
    auto root = this->getOperation();
    ::mlir::IRRewriter builder(&getContext());
    ::mlir::SymbolTableCollection symbolTableCollection;

    // back-propagate targets from RePartitionOps

    ::mlir::Operation *firstOp = nullptr;

    // find first dist-op
    root->walk([&](::mlir::Operation *op) {
      if (::mlir::isa<::imex::dist::DistDialect>(op->getDialect()) ||
          ::mlir::isa<::mlir::mesh::MeshDialect>(op->getDialect())) {
        firstOp = op;
        return ::mlir::WalkResult::interrupt();
      }
      return ::mlir::WalkResult::advance();
    });
    if (!firstOp) {
      return;
    }
    builder.setInsertionPoint(firstOp);

    // find InsertSliceOp and SubviewOp operating on the same base pointer
    // opsGroups holds independent partial operation sequences operating on a
    // specific base pointer
    // on the way compute and back-propagate target parts for InsertSliceOps

    std::unordered_map<::mlir::Operation *,
                       ::mlir::SmallVector<::mlir::Operation *>>
        opsGroups;
    std::unordered_map<::mlir::Operation *, ::mlir::Operation *> baseIPts;

    root->walk([&](::mlir::Operation *op) {
      ::mlir::Value val;
      if (auto typedOp = ::mlir::dyn_cast<::imex::ndarray::InsertSliceOp>(op)) {
        val = typedOp.getDestination();
      } else if (auto typedOp =
                     ::mlir::dyn_cast<::imex::ndarray::SubviewOp>(op)) {
        val = typedOp.getSource();
      }
      if (val) {
        auto base = getBaseArray(val);
        baseIPts.emplace(base, getShardOp(base));
        opsGroups[base].emplace_back(op);

        // for InsertSliceOps compute and propagate target parts
        if (auto typedOp =
                ::mlir::dyn_cast<::imex::ndarray::InsertSliceOp>(op)) {
          builder.setInsertionPointAfter(baseIPts[base]);
          auto srcop =
              typedOp.getSource().getDefiningOp<::mlir::mesh::ShardOp>();
          assert(srcop && "InsertSliceOp must have a ShardOp as source");
          assert(srcop.getAnnotateForUsers());
          auto target = computeTarget(builder, typedOp, srcop.getSharding());
          baseIPts[base] = target;
          typedOp.getTargetMutable().assign(target->getResult(0));
          backPropagateSharding(builder, srcop, target);
        }
      }
    });

    if (!opsGroups.empty()) {
      // outer loop iterates base over base pointers
      for (auto grpP : opsGroups) {
        if (grpP.second.empty())
          continue;
        grpP.second.emplace_back(nullptr);

        auto &base = grpP.first;

        auto shardOp = getShardOp(base);
        builder.setInsertionPointAfter(shardOp);

        // find groups operating on the same base, groups are separated by write
        // operations (InsertSliceOps for now)
        ::mlir::SmallVector<::mlir::DenseI64ArrayAttr> svOffs, svSizes,
            svStrides;
        ::imex::ValVec subviews, constraints;

        for (auto i : grpP.second) {
          // collect SubviewOps until we meet a InsertSliceOp
          if (auto subviewOp =
                  i ? ::mlir::dyn_cast<::imex::ndarray::SubviewOp>(*i)
                    : ::imex::ndarray::SubviewOp()) {
            if (subviewOp && !subviewOp->hasAttr("final")) {
              auto sOffs = subviewOp.getStaticOffsets();
              auto sSizes = subviewOp.getStaticSizes();
              auto sStrides = subviewOp.getStaticStrides();
              assert(
                  !(::mlir::ShapedType::isDynamicShape(sSizes) ||
                    ::mlir::ShapedType::isDynamicShape(sOffs) ||
                    ::mlir::ShapedType::isDynamicShape(sStrides)) ||
                  (false &&
                   "SubviewOp must have dynamic offsets, sizes and strides"));
              auto svShardOp = getShardOp(subviewOp);
              assert(svShardOp);
              auto target = svShardOp.getConstraint();
              assert(target);
              svOffs.emplace_back(builder.getDenseI64ArrayAttr(sOffs));
              svSizes.emplace_back(builder.getDenseI64ArrayAttr(sSizes));
              svStrides.emplace_back(builder.getDenseI64ArrayAttr(sStrides));
              constraints.emplace_back(target);
              subviews.emplace_back(subviewOp.getResult());
            }
          } else {
            // this is no SubViewOp
            auto updateSharding = [&shardOp](::mlir::mesh::ShardOp op,
                                             ::mlir::mesh::ShardOp nOp) {
              // assert(op.getSrc() == shardOp);
              op.getConstraintMutable().assign(nOp.getConstraint());
              op.getSrcMutable().assign(nOp);
            };

            ::mlir::mesh::ShardOp nShardOp;

            // compute bounding box and update ShardOp and SubviewOps
            if (!constraints.empty()) {
              builder.setInsertionPointAfter(baseIPts[base]);
              auto bb = builder.create<::imex::dist::BoundingBoxOp>(
                  shardOp->getLoc(),
                  ::imex::dist::ShardingConstraintType::get(
                      builder.getContext()),
                  ::imex::dist::DistSliceAttr::get(builder.getContext(), svOffs,
                                                   svSizes, svStrides),
                  constraints);
              nShardOp =
                  ::mlir::cast<::mlir::mesh::ShardOp>(builder.clone(*shardOp));
              nShardOp.getConstraintMutable().assign(bb);

              for (auto sv : subviews) {
                auto svOp = sv.getDefiningOp<::imex::ndarray::SubviewOp>();
                auto svShardOp = getShardOpOfOperand(svOp.getSource());
                updateSharding(svShardOp, nShardOp);
              }
              constraints.clear();
              subviews.clear();
              svOffs.clear();
              svSizes.clear();
              svStrides.clear();
              shardOp = nShardOp;
            }

            // update dependent InsertSliceOp
            // we also need an explicit ShardOp after a InsertSliceOp
            if (auto insertOp =
                    i ? ::mlir::dyn_cast<::imex::ndarray::InsertSliceOp>(*i)
                      : ::imex::ndarray::InsertSliceOp()) {
              auto destShard = getShardOpOfOperand(insertOp.getDestination());
              assert(destShard);
              updateSharding(destShard, nShardOp);
              builder.setInsertionPointAfter(insertOp);
              shardOp =
                  ::mlir::cast<::mlir::mesh::ShardOp>(builder.clone(*shardOp));
              builder.setInsertionPointAfter(shardOp);
            }
          }
        } // for (auto j = grpP.second.begin(); j != grpP.second.end(); ++j)

        // we might have created a spurious ShardOp
        if (shardOp && shardOp->use_empty()) {
          builder.eraseOp(shardOp);
        }
      } // for (auto grpP : opsGroups)
    } // if (!shardOps.empty())
#endif // 0
  }
};

} // namespace
} // namespace dist

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::Pass> createDistCoalescePass() {
  return std::make_unique<::imex::dist::DistCoalescePass>();
}

} // namespace imex
