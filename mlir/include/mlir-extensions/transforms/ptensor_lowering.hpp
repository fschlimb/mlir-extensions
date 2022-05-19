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

// Lowering PTensor operations to LinAlg (for now)

#pragma once

#include "mlir-extensions/dialect/ptensor/dialect.hpp"

#include <mlir/IR/PatternMatch.h>

namespace ptensor {

    // Lowering arange to LinAlg
    struct ARangeLowering : public mlir::OpRewritePattern<ptensor::ARangeOp> {
        using OpRewritePattern::OpRewritePattern;

        mlir::LogicalResult
        matchAndRewrite(ptensor::ARangeOp op, mlir::PatternRewriter &rewriter) const override;
    };

    // Lowering element-wise binary operations to LinAlg
    struct EWBinOpLowering : public mlir::OpRewritePattern<ptensor::EWBinOp> {
        using OpRewritePattern::OpRewritePattern;

        mlir::LogicalResult
        matchAndRewrite(ptensor::EWBinOp op, mlir::PatternRewriter &rewriter) const override;

        static mlir::Type getEWBinOpRType(mlir::PatternRewriter &rewriter, EWBinOpId op, const mlir::Type & lhsType, const mlir::Type & rhsType);
    };

} // namespace ptensor