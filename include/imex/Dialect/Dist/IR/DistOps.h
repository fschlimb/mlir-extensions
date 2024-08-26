//===- DistOps.h - Dist dialect  -------------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the Dist dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#ifndef _Dist_OPS_H_INCLUDED_
#define _Dist_OPS_H_INCLUDED_

#include <imex/Utils/PassUtils.h>
#include <mlir/Bytecode/BytecodeOpInterface.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include <iostream>

namespace llvm {
template <typename T>
hash_code hash_value(const SmallVector<SmallVector<T>> &arg) {
  hash_code hval((T)11);
  for (const auto &v : arg) {
    if (v.size())
      hval =
          hash_combine(hval, hash_combine_range(v.data(), v.data() + v.size()));
  }
  return hval;
}
} // namespace llvm

namespace imex {
namespace ndarray {
class NDArrayType;
} // namespace ndarray
namespace dist {
using ::mlir::DenseI64ArrayAttr;
} // namespace dist
} // namespace imex

#include <imex/Dialect/Dist/IR/DistOpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/Dist/IR/DistOpsTypes.h.inc>
#define GET_ATTRDEF_CLASSES
#include <imex/Dialect/Dist/IR/DistOpsAttrs.h.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/Dist/IR/DistOps.h.inc>

#endif // _Dist_OPS_H_INCLUDED_
