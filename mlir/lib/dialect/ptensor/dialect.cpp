// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mlir-extensions/dialect/ptensor/dialect.hpp"

namespace ptensor {

    void PTensorDialect::initialize()
    {
        addOperations<
#define GET_OP_LIST
#include "mlir-extensions/dialect/ptensor/PTensorOps.cpp.inc"
            >();
    }
    
    void ARangeOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state, ::mlir::Value start, ::mlir::Value stop, ::mlir::Value step, bool dist)
    {
        auto dataType = ::mlir::RankedTensorType::get({-1}, builder.getI64Type());
        ARangeOp::build(builder, state, dataType, start, stop, step, dist);
    }

    void EWBinOp::build(::mlir::OpBuilder &builder, ::mlir::OperationState &state, ::mlir::Value lhs, ::mlir::Value rhs, bool dist)
    {
        //auto dataType = lhs.getType().dyn_cast<::mlir::RankedTensorType>();
        //assert(dataType);
        auto dataType = ::mlir::RankedTensorType::get({-1}, builder.getI64Type());
        EWBinOp::build(builder, state, dataType, lhs, rhs, dist);
    }
} // namespace ptensor

#include "mlir-extensions/dialect/ptensor/PTensorOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mlir-extensions/dialect/ptensor/PTensorOps.cpp.inc"
