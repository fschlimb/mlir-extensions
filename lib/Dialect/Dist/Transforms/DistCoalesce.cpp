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
/// This pass tries to minimize the number of dist::RePartitionOps.
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
  ::mlir::Operation *getArray(const ::mlir::Value &val) {
    auto defOp = val.getDefiningOp();
    if (isAnyOf<::imex::ndarray::EWBinOp, ::imex::ndarray::EWUnyOp,
                ::imex::ndarray::ReshapeOp, ::imex::ndarray::CopyOp,
                ::imex::ndarray::LinSpaceOp, ::imex::ndarray::CreateOp>(
            defOp)) {
      return defOp;
    } else if (auto op = ::mlir::dyn_cast<::imex::ndarray::SubviewOp>(defOp)) {
      return getArray(op.getSource());
    } else if (auto op =
                   ::mlir::dyn_cast<::imex::ndarray::InsertSliceOp>(defOp)) {
      return getArray(op.getDestination());
    } else if (auto op = ::mlir::dyn_cast<::imex::dist::ShardOp>(defOp)) {
      return getArray(op.getSrc());
    } else if (auto op = ::mlir::dyn_cast<::mlir::UnrealizedConversionCastOp>(
                   defOp)) {
      if (op.getInputs().size() == 1) {
        return getArray(op.getInputs().front());
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
  bool is_temp(::imex::dist::ShardOp &op) {
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
          return builder.create<::imex::dist::RePartitionOp>(
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
                             ::imex::dist::ShardOp op) {
    ::mlir::Operation *nOp = nullptr;
    auto target = op.getConstraint();
    if (!target) {
      return;
    }
    auto defOp = op.getSrc().getDefiningOp();
    if (defOp) {
      backPropagateSharding(builder, defOp, target, nOp);
      assert(nOp == nullptr);
    }
    return;
  }

  /// The actual back propagation of target parts
  /// if meeting a supported op, recursively gets defining ops and back
  /// propagates as it follows only supported ops, all other ops act as
  /// propagation barriers (e.g. InsertSliceOps) on the way it updates target
  /// info on SubviewOps and marks RePartitionOps for elimination
  void backPropagateSharding(::mlir::IRRewriter &builder, ::mlir::Operation *op,
                             ::mlir::Value target, ::mlir::Operation *&nOp) {
    nOp = nullptr;
    if (auto typedOp = ::mlir::dyn_cast<::imex::dist::ShardOp>(op)) {
      assert(!typedOp.getConstraint() || (false && "not implemented yet"));
      typedOp.getConstraintMutable().assign(target);
      op = typedOp.getSrc().getDefiningOp();
      assert(op);
    }
    if (auto typedOp = ::mlir::dyn_cast<::imex::ndarray::EWBinOp>(op)) {
      op = typedOp.getLhs().getDefiningOp();
      if (op) {
        backPropagateSharding(builder, op, target, nOp);
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
    } else if (!::mlir::isa<::imex::dist::ShardOp>(op)) {
      op = nullptr;
    }
    if (op) {
      backPropagateSharding(builder, op, target, nOp);
    }
    return;
  }

  ::imex::dist::ShardOp getShardOp(::mlir::Operation *op) {
    assert(op->hasOneUse() || op->use_empty());
    op = *op->user_begin();
    if (::mlir::isa<::mlir::UnrealizedConversionCastOp>(op)) {
      assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
      assert(op->hasOneUse());
      op = *op->user_begin();
    }
    return ::mlir::dyn_cast<::imex::dist::ShardOp>(op);
  }

  ::imex::dist::ShardOp getShardOpOfVal(::mlir::Value val) {
    auto op = val.getDefiningOp();
    if (::mlir::isa<::mlir::UnrealizedConversionCastOp>(op)) {
      assert(op->getNumOperands() == 1 && op->getNumResults() == 1);
      assert(op->hasOneUse() && op->getNumOperands() == 1);
      op = op->getOperand(0).getDefiningOp();
    }
    assert(op->hasOneUse());
    return ::mlir::dyn_cast<::imex::dist::ShardOp>(op);
  }

  /// compute target part for a given InsertSliceOp
  ::mlir::Operation *computeTarget(::mlir::IRRewriter &builder,
                                   ::imex::ndarray::InsertSliceOp op) {
    auto sOffs = op.getStaticOffsets();
    auto sSizes = op.getStaticSizes();
    auto sStrides = op.getStaticStrides();
    assert(!(::mlir::ShapedType::isDynamicShape(sSizes) ||
             ::mlir::ShapedType::isDynamicShape(sOffs) ||
             ::mlir::ShapedType::isDynamicShape(sStrides)) ||
           (false && "SubviewOp must have dynamic offsets, sizes and strides"));

    auto src = getShardOpOfVal(op.getDestination()).getSrc();
    return builder.create<::imex::dist::TargetOfSliceOp>(
        op->getLoc(),
        ::imex::dist::ShardingConstraintType::get(builder.getContext()), src,
        sOffs, sSizes, sStrides);
  }

  //       shardOp = getShardOpOfVal(dest); builder.getI64ArrayAttr(sOffs),
  //       builder.getI64ArrayAttr(sSizes), builder.getI64ArrayAttr(sStrides));
  //   auto dest = op.getDestination();
  //   auto destType =
  //       ::mlir::cast<::mlir::ShapedType>(dest.getType());
  //   assert(destType.hasStaticShape());
  //   auto tShape = destType.getShape();

  //   auto shardOp = getShardOpOfVal(dest);
  //   auto meshOp = mlir::mesh::getMesh(op, shardOp.getShard().getMesh(),
  //   symbolTableCollection); auto meshShape = meshOp.getShape(); auto nDevices
  //   = ::mlir::ShapedType::getNumElements(meshShape); // will assert if not
  //   static auto sAxes = shardOp.getShard().getSplitAxes(); auto tRank =
  //   tShape.size(); auto sRank = sAxes.size(); auto mRank = meshShape.size();
  //   assert(tRank >= sRank);

  //   ::mlir::SmallVector<::mlir::SmallVector<int64_t>> targetSzs;
  //   ::mlir::SmallVector<int64_t> meshSzs;
  //   ::mlir::SmallVector<int64_t> meshStrides(t1);

  //   for (auto [i, tshape] : llvm::enumerate(sAxes)) {
  //     auto sz = tShape[i];
  //     int64_t sz = 1;
  //     if (i<sRank) {
  //       assert(sAxes[i].size() > 0 || (false && "replication axes not
  //       supported")); for (auto j : sAxes[i].asArrayRef()) {
  //         assert(j < (int64_t)mRank);
  //         sz *= meshShape[j];
  //       }
  //     }
  //     meshSzs.emplace_back(sz);
  //   }

  //   targetSzs.resize(sRank);
  //   for (auto dev = 0u; dev<nDevices; ++dev) { // loop over devices
  //     auto idx = ::mlir::
  //     // compute tile size and local size (which can be greater)
  //     auto rem = sz % splitSz;
  //     auto tSz = sz / splitSz;
  //     auto lSz = tSz + ((dev + rem) >= splitSz ? 1l : 0l);
  //     // auto lOff = (dev * tSz) + std::max(0l, rem - (splitSz - dev));
  //     targetSzs[dev].emplace_back(lSz);
  //     }
  //   }

  //   return DistTargetAttr::get(builder.getContext(), targetSzs);
  // }

  // This pass tries to combine multiple RePartitionOps into one.
  // Dependent operations (like SubviewOp) get adequately annotated.
  //
  // The basic idea is to compute a the bounding box of several RePartitionOps
  // and use it for a single repartition. Dependent SubviewOps can then
  // extract the appropriate part from that bounding box without further
  // communication/repartitioning.
  //
  // 1. back-propagation of explicit target-parts
  // 2. group and move SubviewOps
  // 3. create base RePartitionOps and update dependent SubviewOps
  void runOnOperation() override {

    auto root = this->getOperation();
    ::mlir::IRRewriter builder(&getContext());
    ::mlir::SymbolTableCollection symbolTableCollection;

    // back-propagate targets from RePartitionOps

    ::std::set<::imex::dist::RePartitionOp> shardOpsToElimNew;
    ::mlir::SmallVector<::imex::dist::RePartitionOp> rpOps;
    ::mlir::SmallVector<::mlir::func::ReturnOp> retOps;
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

    // insert temporary casts for block args so that we have a base operation
    ::mlir::SmallVector<::mlir::UnrealizedConversionCastOp> dummyCasts;
    for (::mlir::Block &block : root) {
      for (::mlir::BlockArgument &arg : block.getArguments()) {
        if (isDist(arg) && !arg.use_empty()) {
          auto op = builder.create<::mlir::UnrealizedConversionCastOp>(
              builder.getUnknownLoc(), arg.getType(), arg);
          arg.replaceAllUsesExcept(op.getResult(0), op);
          dummyCasts.emplace_back(op);
        }
      }
    }

    // target-annotate all insert_slice ops' input operands
    // root->walk([&](::imex::ndarray::InsertSliceOp op) {
    //   auto srcop = op.getSource().getDefiningOp<::imex::dist::ShardOp>();
    //   assert(srcop && "InsertSliceOp must have a ShardOp as source");

    //   auto sOffs = op.getStaticOffsets();
    //   auto sSizes = op.getStaticSizes();
    //   auto sStrides = op.getStaticStrides();
    //   assert(
    //       !(::mlir::ShapedType::isDynamicShape(sSizes) ||
    //         ::mlir::ShapedType::isDynamicShape(sOffs) ||
    //         ::mlir::ShapedType::isDynamicShape(sStrides)) ||
    //       (false && "SubviewOp must have dynamic offsets, sizes and
    //       strides"));
    //   auto dstType =
    //       ::mlir::cast<::mlir::ShapedType>(op.getDestination().getType());
    //   assert(dstType.hasStaticShape());
    //   auto sShape = dstType.getShape();
    //   auto target = DistSliceAttr::get(builder.getContext(),
    //                                    builder.getDenseI64ArrayAttr(sOffs),
    //                                    builder.getDenseI64ArrayAttr(sSizes),
    //                                    builder.getDenseI64ArrayAttr(sStrides),
    //                                    builder.getDenseI64ArrayAttr(sShape));
    //   srcop->setAttr("target", target);
    //   shardOps.emplace_back(srcop);
    // });

    // Find returnops and protect SubviewOp operands
    // FIXME: The implementation doesn't seem to support subviews of subviews
    // FIXME: elimination of temp RePartitionOps should be done in a different
    // pass root->walk([&](::mlir::Operation *op) {
    //   // if (auto typedOp = ::mlir::dyn_cast<::imex::dist::ShardOp>(op)) {
    //   //   if (!typedOp->hasAttr("target")) {
    //   //     if (is_temp(typedOp)) {
    //   //       shardOpsToElimNew.emplace(typedOp);
    //   //     }
    //   //   }
    //   // } else
    //   if (auto typedOp = ::mlir::dyn_cast<::mlir::func::ReturnOp>(op)) {
    //     retOps.emplace_back(typedOp);
    //   }
    // });

    // for (auto retOp : retOps) {
    //   backPropagateReturn(builder, retOp);
    // }
    // retOps.clear();

    // auto &dom = this->getAnalysis<::mlir::DominanceInfo>();

    // eliminate no longer needed RePartitionOps
    // for (auto rp : shardOpsToElimNew) {
    //   builder.replaceOp(rp, rp.getArray());
    // }

    // find InsertSliceOp, SubviewOp and RePartitionOps on the same base
    // pointer
    // opsGroups holds independent partial operation sequences operating on a
    // specific base pointer

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
        auto base = getArray(val);
        baseIPts.emplace(base, getShardOp(base));
        opsGroups[base].emplace_back(op);
        if (auto typedOp =
                ::mlir::dyn_cast<::imex::ndarray::InsertSliceOp>(op)) {
          builder.setInsertionPointAfter(baseIPts[base]);
          auto target = computeTarget(builder, typedOp);
          baseIPts[base] = target;
          auto srcop =
              typedOp.getSource().getDefiningOp<::imex::dist::ShardOp>();
          assert(srcop && "InsertSliceOp must have a ShardOp as source");
          srcop.getConstraintMutable().assign(target->getResult(0));
          backPropagateSharding(builder, srcop);
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
        // auto &dom = this->getAnalysis<::mlir::DominanceInfo>();

        auto shardOp = getShardOp(base);
        builder.setInsertionPointAfter(shardOp);
        // auto baseShard = shardOp.getShard();

        // find groups operating on the same base, groups are separated by write
        // operations (InsertSliceOps for now)
        // for (auto j = grpP.second.begin(); j != grpP.second.end(); ++j) {
        ::mlir::ValueRange bbOffs, bbSizes;
        ::mlir::SmallVector<::mlir::DenseI64ArrayAttr> bbOffsVec, bbSizesVec,
            bbStridesVec, tOffsVec, tSizesVec, tStridesVec, tBaseSizesVec;
        ::imex::ValVec subviews;
        ::imex::ValVec constraints;

        for (auto i : grpP.second) {
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
              bbOffsVec.emplace_back(builder.getDenseI64ArrayAttr(sOffs));
              bbSizesVec.emplace_back(builder.getDenseI64ArrayAttr(sSizes));
              bbStridesVec.emplace_back(builder.getDenseI64ArrayAttr(sStrides));
              constraints.emplace_back(target);
              // tOffsVec.emplace_back(target.getSlcOffsets().front());
              // tSizesVec.emplace_back(target.getSlcSizes().front());
              // tStridesVec.emplace_back(target.getSlcStrides().front());
              // tBaseSizesVec.emplace_back(target.getBaseSizes().front());
              subviews.emplace_back(subviewOp.getResult());
            }
          } else {
            auto updateSharding = [&shardOp](::imex::dist::ShardOp op,
                                             ::imex::dist::ShardOp nOp) {
              // assert(op.getSrc() == shardOp);
              op.getConstraintMutable().assign(nOp.getConstraint());
              op.getSrcMutable().assign(nOp);
            };

            ::imex::dist::ShardOp nShardOp;

            if (!constraints.empty()) {
              builder.setInsertionPointAfter(baseIPts[base]);
              auto bb = builder.create<::imex::dist::BoundingBoxOp>(
                  shardOp->getLoc(),
                  ::imex::dist::ShardingConstraintType::get(
                      builder.getContext()),
                  ::imex::dist::DistSliceAttr::get(builder.getContext(),
                                                   bbOffsVec, bbSizesVec,
                                                   bbStridesVec),
                  constraints);
              nShardOp =
                  ::mlir::cast<::imex::dist::ShardOp>(builder.clone(*shardOp));
              nShardOp.getConstraintMutable().assign(bb);

              for (auto sv : subviews) {
                auto svOp = sv.getDefiningOp<::imex::ndarray::SubviewOp>();
                auto svShardOp = getShardOpOfVal(svOp.getSource());
                updateSharding(svShardOp, nShardOp);
              }
              constraints.clear();
              subviews.clear();
              bbOffsVec.clear();
              bbSizesVec.clear();
              bbStridesVec.clear();
              shardOp = nShardOp;
            }

            if (auto insertOp =
                    i ? ::mlir::dyn_cast<::imex::ndarray::InsertSliceOp>(*i)
                      : ::imex::ndarray::InsertSliceOp()) {
              auto destShard = getShardOpOfVal(insertOp.getDestination());
              assert(destShard);
              updateSharding(destShard, nShardOp);
              builder.setInsertionPointAfter(insertOp);
              shardOp =
                  ::mlir::cast<::imex::dist::ShardOp>(builder.clone(*shardOp));
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

    // Get rid of dummy casts
    for (auto op : dummyCasts) {
      op.getResult(0).replaceAllUsesWith(op->getOperand(0));
      builder.eraseOp(op);
    }
  }
};

#if 0
          ::mlir::SmallVector<::imex::dist::RePartitionOp> shardOpsToElim;
          auto fOp = grp.front();
            ::mlir::Operation *eIPnt = nullptr;
            auto rpIPnt = fOp;
            auto bbIPnt = fOp;
            ::mlir::ValueRange bbOffs, bbSizes;
            ::mlir::Operation *combined = nullptr;

            // iterate group
            for (auto i = grp.begin(); i != grp.end(); ++i) {
              auto e = ::mlir::dyn_cast<::imex::dist::SubviewOp>(*i);
              if (e && e->hasAttr("final")) {
                continue;
              }
              auto rp = ::mlir::dyn_cast<::imex::dist::RePartitionOp>(*i);
              // check if we can move current op up
              if (dom.dominates(fOp, *i)) {
                bool can_move = true;
                if (fOp != *i) {
                  for (auto o : (*i)->getOperands()) {
                    if (!dom.dominates(o.getDefiningOp(),
                                       eIPnt ? eIPnt : rpIPnt)) {
                      can_move = false;
                      break;
                    }
                  }
                }

                // if it's safe to move: do it
                if (can_move) {
                  if (e) {
                    if (false && nSubViews < 2) {
                      e->moveBefore(rpIPnt);
                    } else {
                      auto loc = e.getLoc();
                      builder.setInsertionPointAfter(bbIPnt);

                      auto _sizes = getMixedAsValues(loc, builder, e.getSizes(),
                                                     e.getStaticSizes());
                      auto _offs = getMixedAsValues(
                          loc, builder, e.getOffsets(), e.getStaticOffsets());
                      auto _strides = getMixedAsValues(
                          loc, builder, e.getStrides(), e.getStaticStrides());

                      // if there is no target part, compute/use default
                      ::mlir::ValueRange tOffs = e.getTargetOffsets();
                      ::mlir::ValueRange tSizes = e.getTargetSizes();
                      if (tOffs.empty()) {
                        assert(tSizes.empty());
                        auto defPart =
                            builder.create<::imex::dist::DefaultPartitionOp>(
                                loc, nProcs, pRank, _sizes);
                        tOffs = defPart.getLOffsets();
                        tSizes = defPart.getLShape();
                        bbIPnt = defPart;
                        auto nop = updateTargetPart(builder, e, tOffs, tSizes);
                        assert(!nop);
                      }

                      // compute/extend local bounding box
                      auto bbox =
                          builder.create<::imex::dist::LocalBoundingBoxOp>(
                              loc, false, _offs, _sizes, _strides, tOffs,
                              tSizes, bbOffs, bbSizes);
                      bbOffs = bbox.getResultOffsets();
                      bbSizes = bbox.getResultSizes();
                      bbIPnt = bbox;
                      assert(bbOffs.size() == bbSizes.size());

                      // make BB available to repartitionop
                      if (combined) {
                        auto rank = bbOffs.size();
                        combined->setOperands(1 + 0 * rank, rank, bbOffs);
                        combined->setOperands(1 + 1 * rank, rank, bbSizes);
                      } else {
                        for (auto o : bbOffs) {
                          assert(dom.dominates(o.getDefiningOp(), bbIPnt));
                        }
                        for (auto o : bbSizes) {
                          assert(dom.dominates(o.getDefiningOp(), bbIPnt));
                        }
                        auto dynPtType = cloneWithDynEnv(
                            mlir::cast<::imex::ndarray::NDArrayType>(
                                base->getResult(0).getType()));
                        combined = builder.create<::imex::dist::RePartitionOp>(
                            loc, dynPtType, base->getResult(0), bbOffs,
                            bbSizes);
                      }
                      e->moveAfter(eIPnt ? eIPnt : combined);
                      e->setOperand(0, combined->getResult(0));
                      eIPnt = *i;
                      // any RepartitionOps of this extract slice can
                      // potentially be eliminated
                      for (auto u : e->getUsers()) {
                        if (auto r =
                                ::mlir::dyn_cast<::imex::dist::RePartitionOp>(
                                    u)) {
                          shardOpsToElim.emplace_back(u);
                        }
                      }
                    }
                  } else { // if e
                    assert(rp);
                    (*i)->moveAfter(rpIPnt);
                    rpIPnt = *i;
                  }
                  continue;
                } // if(can_move)
              }   // dominates
              // if fOp does not dominate i or i's inputs do not dominate fOp
              // we try later with remaining unhandled ops
              unhandled.emplace_back(*i);
            } // for grp

            // FIXME: handling of remaining repartitionops needs simplification
            for (auto o : shardOpsToElim) {
              for (auto x : grp) {
                // elmiminate only if it is in our current group
                if (x == o) {
                  assert(o.getTargetOffsets().empty());
                  // remove from unhandled
                  for (auto it = unhandled.begin(); it != unhandled.end();
                       ++it) {
                    if (*it == o) {
                      unhandled.erase(it);
                      break;
                    }
                  }
                  builder.replaceOp(o, o.getArray());
                  break;
                }
              }
            }
            grp.clear();
            grp.swap(unhandled);
          }
          if (j == grpP.second.end()) {
            break;
          } // while (grp.size() > 0)
        }   // for (auto j = grpP.second.begin(); j != grpP.second.end(); ++j)
      }     // for (auto grpP : opsGroups)
    }       // !rpOps.empty()
#endif // if 0

} // namespace
} // namespace dist

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::Pass> createDistCoalescePass() {
  return std::make_unique<::imex::dist::DistCoalescePass>();
}

} // namespace imex
