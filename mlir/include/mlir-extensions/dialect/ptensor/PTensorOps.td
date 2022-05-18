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

#ifndef _PTENSOR_OPS_INCLUDED_
#define _PTENSOR_OPS_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'PTensor' dialect in the ODS framework so that we
// can define our operations.
def PTensor_Dialect : Dialect {
    // The namespace of our dialect
    let name = "ptensor";
    
    // A short one-line summary of our dialect.
    let summary = "A high-level dialect for parallel tensor operations";
    
    // A longer description of our dialect.
    let description = [{
            The ptensor dialect describe parallel operations on tensors.
            Generic parallel patterns are provided, such as element-wise-unary,
            element-wise-binary or reduce.
        }];
    
    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::ptensor";
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class PTensor_Op<string mnemonic, list<Trait> traits = []> :
    Op<PTensor_Dialect, mnemonic, traits>;


def ARangeOp : PTensor_Op<"arange", [NoSideEffect]> {
    // arange takes 3 integer attributes (for now): start, stop and step
    let arguments = (ins AnyType : $start, AnyType : $stop, AnyType : $step, BoolAttr : $dist);

    // result is a tensor
    let results = (outs AnyType);

    // Trivial builder
    let builders = [OpBuilder<(ins "::mlir::Value" : $start, "::mlir::Value" : $stop, "::mlir::Value" : $step, "bool" : $dist)>];
}

def EWBinOp : PTensor_Op<"ewbin", [NoSideEffect]> {
    // arange takes 3 integer attributes (for now): start, stop and step
    let arguments = (ins AnyType : $lhs, AnyType : $rhs, BoolAttr : $dist);

    // result is a tensor
    let results = (outs AnyType);
    
    // Trivial builder
    let builders = [OpBuilder<(ins "::mlir::Value" : $lhs, "::mlir::Value" : $rhs, "bool" : $dist)>];
}

#endif // _PTENSOR_OPS_INCLUDED_
