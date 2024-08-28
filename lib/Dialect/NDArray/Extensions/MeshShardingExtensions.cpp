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
// Computes result sharding by extending a copy of the input sharding with shard sizes.
// The shard sizes reflect the sizes resulting from the non-copying subview operation.
// Requires sharding of input tensor.
FailureOr<MeshSharding> getShardedDimsSizes(Value ary, OffsetSizeAndStrideOpInterface op) {
  SymbolTableCollection symbolTable;
  auto aryType = cast<RankedTensorType>(ary.getType());
  // currently no support for dynamic input shapes
  if (!aryType.hasStaticShape())
    return failure();
  auto aryShape = aryType.getShape();
  auto rank = cast<RankedTensorType>(ary.getType()).getRank();

  auto aryShardOp = ary.getDefiningOp<mesh::ShardOp>();
  // currently no support for non-sharded source
  if (!aryShardOp)
    return failure();

  auto offs = op.getStaticOffsets();
  auto sizes = op.getStaticSizes();
  auto strides = op.getStaticStrides();
  // currently no support for dynamic subviews
  if (ShapedType::isDynamicShape(offs) ||
      ShapedType::isDynamicShape(sizes) ||
      ShapedType::isDynamicShape(strides))
    return failure();

  auto arySharding = aryShardOp.getSharding().getDefiningOp<mesh::ShardingOp>();
  // currently no support for sharding dims sizes on input
  if(!arySharding.getStaticShardedDimsSizes().empty())
    return failure();

  auto mesh = getMesh(arySharding, symbolTable);
  if (!mesh)
    return failure();
  auto meshShape = mesh.getShape();
  // currently no support for dynamic mesh shape
  if (llvm::any_of(meshShape, [](int64_t dim) { return dim == ShapedType::kDynamic; }))
    return failure();

  auto splitAxes = arySharding.getSplitAxes();
  assert((int64_t)splitAxes.size() <= rank);
  
  // flattened shard sizes for each dimension (see sharding.sharded_dims_sizes) after subview
  SmallVector<int64_t> splitSzs;
  // iterate split tensor dimensions
  for(auto dim = 0u; dim<splitAxes.size(); ++dim) {
    auto axes = arySharding.getSplitAxes().getAxes()[dim].asArrayRef();
    if (axes.empty()) continue;
    int64_t splitSz = 1;  // number of shards in this dimension
    for(auto i : axes) splitSz *= meshShape[i];
    int64_t shardSz = shardDimension(aryShape[dim], splitSz);
    int64_t pos = offs[dim]; // current position in split tensor dim
    int64_t mx = sizes[dim]; // max number of elements we assign to current shard
    for(auto shard = 0; shard < splitSz; ++shard) {
      // extract size of overlap of subview with current input shard
      auto shardStart = shard * shardSz;
      auto shardEnd = shardStart + shardSz;
      auto num = shardEnd - pos;
      int64_t sz = 0;
      if (num > 0) { // if starts before end of shard
        sz = (num + (strides[dim]-1)) / strides[dim];
        sz = std::min(mx, sz);
      }
      splitSzs.emplace_back(sz);
      // update pos and max for next result shard
      pos += sz * strides[dim];
      mx -= sz;
    }
  }
  return MeshSharding::get(arySharding.getMeshAttr(),
                            arySharding.getSplitAxes().getAxes(),
                            arySharding.getPartialAxes().value_or(llvm::ArrayRef<MeshAxis>{}),
                            arySharding.getPartialType().value_or(ReductionKind::Sum),
                            {}, // static halo
                            splitSzs,
                            {},
                            {});
}

// Sharding of tensor.empty
template<typename T, typename OpType>
struct OffsetSizeAndStrideShardingInterface
    :  public ShardingInterface::ExternalModel<T, OpType> {
  using parent = ShardingInterface::ExternalModel<T, OpType>;
  
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

  LogicalResult spmdize(::mlir::Operation *op, ArrayRef<Value> spmdizedOperands, ArrayRef<MeshSharding> operandShardings, ArrayRef<MeshSharding> resultShardings, IRMapping&spmdizationMap, SymbolTableCollection &symbolTableCollection, OpBuilder &builder) const {
    LLVM_DEBUG(DBGS() << "spmdize\n");
    spmdizeTriviallyShardableOperation(*op, spmdizedOperands, operandShardings,
                                              resultShardings, spmdizationMap,
                                              symbolTableCollection, builder);
    return success();
  }
};

struct SubviewShardingInterface : public OffsetSizeAndStrideShardingInterface<SubviewShardingInterface, imex::ndarray::SubviewOp> {
  LogicalResult addShardingAnnotations(::mlir::Operation *op, OpBuilder &b, const ShardingOption &shardingOption) const {
    LLVM_DEBUG(DBGS() << "addShardingAnnotations\n");
    auto svop = cast<SubviewOp>(op);
    auto sharding = getShardedDimsSizes(svop.getSource(), svop);
    if(failed(sharding)) return failure();
    maybeInsertTargetShardingAnnotation(sharding.value(), op->getResult(0), b);

    return success();
  }
};

struct InsertSliceShardingInterface : public OffsetSizeAndStrideShardingInterface<InsertSliceShardingInterface, imex::ndarray::InsertSliceOp> {
  LogicalResult addShardingAnnotations(::mlir::Operation *op, OpBuilder &b, const ShardingOption &shardingOption) const {
    LLVM_DEBUG(DBGS() << "addShardingAnnotations\n");
    auto svop = cast<InsertSliceOp>(op);
    auto sharding = getShardedDimsSizes(svop.getDestination(), svop);
    if(failed(sharding)) return failure();
    maybeInsertSourceShardingAnnotation(sharding.value(), op->getOpOperand(1), b);

    return success();
  }
};
} // namespace

void imex::ndarray::registerShardingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, imex::ndarray::NDArrayDialect *dialect) {
    SubviewOp::template attachInterface<SubviewShardingInterface>(*ctx);
    InsertSliceOp::template attachInterface<InsertSliceShardingInterface>(*ctx);
  });
}
