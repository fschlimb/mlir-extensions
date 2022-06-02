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

#include <mlir-extensions/dialect/ptensor/dialect.hpp>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>

namespace ptensor {

    void PTensorDialect::initialize()
    {
        addTypes<
#define GET_TYPEDEF_LIST
#include "mlir-extensions/dialect/ptensor/PTensorOpsTypes.cpp.inc"
            >();
        addOperations<
#define GET_OP_LIST
#include "mlir-extensions/dialect/ptensor/PTensorOps.cpp.inc"
            >();
    }

} // namespace ptensor

#include "mlir-extensions/dialect/ptensor/PTensorOpsDialect.cpp.inc"
#define GET_TYPEDEF_CLASSES
#include "mlir-extensions/dialect/ptensor/PTensorOpsTypes.cpp.inc"
#define GET_OP_CLASSES
#include "mlir-extensions/dialect/ptensor/PTensorOps.cpp.inc"
