//===- ShardingInterfaceImpl.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h"
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterfaceImpl.h"
#include "imex/Dialect/NDArray/IR/NDArrayOps.h"
#include "imex/Dialect/NDArray/Extensions/MeshShardingExtensions.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/Debug.h"
// #include <iostream>

#define DEBUG_TYPE "ndarray-sharding-impl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")
// #define DBGS() (std::cerr << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace imex::ndarray;
using namespace mlir::mesh;

namespace {

// Sharding of tensor.empty
struct SubViewShardingInterface
    :  public ShardingInterface::ExternalModel<SubViewShardingInterface, imex::ndarray::SubviewOp> {
  using parent = ShardingInterface::ExternalModel<SubViewShardingInterface, imex::ndarray::SubviewOp>;
  
  SmallVector<mlir::utils::IteratorType> getLoopIteratorTypes(::mlir::Operation *op) const {
    LLVM_DEBUG(DBGS() << "getLoopIteratorTypes\n");
    Value val = op->getOperand(0);
    auto type = dyn_cast<RankedTensorType>(val.getType());
    if (!type)
      return {};
    SmallVector<utils::IteratorType> types(type.getRank(),
                                          utils::IteratorType::parallel);
    return types;
  }
  
  SmallVector<ReductionKind> getReductionLoopIteratorKinds(::mlir::Operation *op) const
  {
    LLVM_DEBUG(DBGS() << "getReductionLoopIteratorKinds\n");
    return parent::getReductionLoopIteratorKinds(op);
  }

  SmallVector<AffineMap> getIndexingMaps(::mlir::Operation *op) const {
    LLVM_DEBUG(DBGS() << "getIndexingMaps\n");
    MLIRContext *ctx = op->getContext();
    Value val = op->getOperand(0);
    auto type = dyn_cast<RankedTensorType>(val.getType());
    if (!type)
      return {};
    int64_t rank = type.getRank();
    int64_t num = op->getNumOperands() + op->getNumResults();
    SmallVector<AffineMap> maps(num,
                                AffineMap::getMultiDimIdentityMap(rank, ctx));
    return maps;
  }

  FailureOr<ShardingOption> getShardingOption(::mlir::Operation *op, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings) const {
    LLVM_DEBUG(DBGS() << "getShardingOption\n");
    return parent::getShardingOption(op, operandShardings, resultShardings);
  }

  FailureOr<std::vector<MeshSharding>> getShardingAnnotations(::mlir::Operation *op, const ShardingOption &shardingOption) const {
    LLVM_DEBUG(DBGS() << "getShardingAnnotations\n");
    return parent::getShardingAnnotations(op, shardingOption);
  }

  LogicalResult addShardingAnnotations(::mlir::Operation *op, OpBuilder &b, const ShardingOption &shardingOption) const {
    LLVM_DEBUG(DBGS() << "addShardingAnnotations\n");
    // return parent::addShardingAnnotations(op, b, shardingOption);
    SymbolTableCollection symbolTable;
    auto svop = cast<SubviewOp>(op);
    auto ary = svop.getSource();
    auto aryType = cast<RankedTensorType>(ary.getType());
    if (!aryType.hasStaticShape())
      return failure(); // currently no support for dynamic input shapes
    auto aryShape = aryType.getShape();
    auto rank = cast<RankedTensorType>(ary.getType()).getRank();

    auto aryShardOp = ary.getDefiningOp<mesh::ShardOp>();
    if (!aryShardOp)
      return failure(); // currently no support for non-sharded source

    auto offs = svop.getStaticOffsets();
    auto sizes = svop.getStaticSizes();
    auto strides = svop.getStaticStrides();
    if (ShapedType::isDynamicShape(offs) ||
        ShapedType::isDynamicShape(sizes) ||
        ShapedType::isDynamicShape(strides))
      return failure(); // currently no support for dynamic subviews

    auto arySharding = aryShardOp.getSharding().getDefiningOp<mesh::ShardingOp>();
    if(!arySharding.getStaticShardedDimsSizes().empty())
      return failure(); // currently no support for sharding dims sizes on input

    auto mesh = getMesh(arySharding, symbolTable);
    if (!mesh)
      return failure();
    auto meshShape = mesh.getShape();
    if (llvm::any_of(meshShape, [](int64_t dim) { return dim == ShapedType::kDynamic; }))
      return failure();
    
    SmallVector<int64_t> meshTileSzs(meshShape.size(), 1);
    for (int64_t dim = 0; dim<(int64_t)meshTileSzs.size(); ++dim) {
      for (auto dim=(int64_t)meshTileSzs.size()-2; dim>=0; --dim) {
        meshTileSzs[dim] = meshShape[dim] * meshTileSzs[dim+1];
      }
    }

    SmallVector<bool> definedAxis(meshShape.size(), false);
    auto splitAxes = arySharding.getSplitAxes();
    assert((int64_t)splitAxes.size() <= rank);
    auto numSplitAxes = 0;
    for (auto [d, axes] : llvm::enumerate(splitAxes)) {
      if (axes.size()) {
        ++numSplitAxes;
        definedAxis[d] = true;
      }
    }
    
    SmallVector<int64_t> splitSzs;
    for(auto dim = 0u; dim<splitAxes.size(); ++dim) {
      auto axes = arySharding.getSplitAxes().getAxes()[dim].asArrayRef();
      if (axes.empty()) continue;
      int64_t splitSz = 1;
      for(auto i : axes) splitSz *= meshShape[i];
      int64_t shardSz = shardDimension(aryShape[dim], splitSz);
      int64_t pos = offs[dim];
      int64_t mx = sizes[dim];
      for(auto shard = 0; shard < splitSz; ++shard) {
        auto shardStart = shard * shardSz;
        auto shardEnd = shardStart + shardSz;
        auto num = shardEnd - pos;
        int64_t sz = 0;
        if (num > 0) {
          sz = (num + (strides[dim]-1)) / strides[dim];
          sz = std::min(mx, sz);
        }
        splitSzs.emplace_back(sz);
        pos += sz * strides[dim];
        mx -= sz;
      }
    }

    auto resSharding = MeshSharding::get(arySharding.getMeshAttr(),
                                         arySharding.getSplitAxes().getAxes(),
                                         arySharding.getPartialAxes().value_or(llvm::ArrayRef<MeshAxis>{}),
                                         arySharding.getPartialType().value_or(ReductionKind::Sum),
                                         {}, // static halo
                                         splitSzs,
                                         {},
                                         {});
    maybeInsertTargetShardingAnnotation(resSharding, op->getResult(0), b);

    return success();
  }

  LogicalResult spmdize(::mlir::Operation *op, ArrayRef<Value> spmdizedOperands, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings, IRMapping&spmdizationMap, SymbolTableCollection &symbolTableCollection, OpBuilder &builder) const {
    LLVM_DEBUG(DBGS() << "spmdize\n");
    spmdizeTriviallyShardableOperation(*op, spmdizedOperands, operandShardings,
                                              resultShardings, spmdizationMap,
                                              symbolTableCollection, builder);
    return success();
  }
};
} // namespace

void imex::ndarray::registerShardingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, imex::ndarray::NDArrayDialect *dialect) {
    SubviewOp::template attachInterface<SubViewShardingInterface>(*ctx);
  });
}
