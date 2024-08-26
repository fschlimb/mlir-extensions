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

#define DEBUG_TYPE "ndarray-sharding-impl"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

using namespace mlir;
using namespace imex::ndarray;
using namespace mlir::mesh;

namespace {

// Sharding of tensor.empty
struct SubViewShardingInterface
    : public ElementwiseShardingInterface<imex::ndarray::SubviewOp> {
};
} // namespace

void imex::ndarray::registerShardingInterfaceExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, imex::ndarray::NDArrayDialect *dialect) {
    SubviewOp::template attachInterface<SubViewShardingInterface>(*ctx);
  });
}
