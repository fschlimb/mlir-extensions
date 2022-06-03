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

#pragma once

#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>

namespace ptensor {

// The set of supported operations
enum EWBinOpId : int {
    ADD,
    AND,
    ATAN2,
    BITWISE_AND,
    BITWISE_LEFT_SHIFT,
    BITWISE_OR,
    BITWISE_RIGHT_SHIFT,
    BITWISE_XOR,
    EQUAL,
    FLOOR_DIVIDE,
    GREATER,
    GREATER_EQUAL,
    LESS,
    LESS_EQUAL,
    LOGADDEXP,
    LOGICAL_AND,
    LOGICAL_OR,
    LOGICAL_XOR,
    LSHIFT,
    MATMUL,
    MAXIMUM,
    MINIMUM,
    MODULO,
    MULTIPLY,
    NOT_EQUAL,
    OR,
    POWER,
    SUBTRACT,
    TRUE_DIVIDE,
    XOR,
    EWBINOPID_LAST
};

enum ReduceOpId : int {
    MAX,
    MEAN,
    MIN,
    PROD,
    SUM,
    STD,
    VAR,
    REDUCEOPID_LAST
};

}

#include <mlir-extensions/Dialect/ptensor/PTensorOpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <mlir-extensions/Dialect/ptensor/PTensorOpsTypes.h.inc>
#define GET_OP_CLASSES
#include <mlir-extensions/Dialect/ptensor/PTensorOps.h.inc>
