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

#include "mlir-extensions/dialect/plier/dialect.hpp"
#include "mlir-extensions/dialect/ptensor/dialect.hpp"

#include <mlir/Transforms/DialectConversion.h>
#include <mlir/IR/PatternMatch.h>

namespace ptensor {

    struct FromNumpyCall : public ::mlir::OpConversionPattern<::plier::PyCallOp>
    {
        using OpConversionPattern::OpConversionPattern;

        ::mlir::LogicalResult
              matchAndRewrite(::plier::PyCallOp op,
                              ::plier::PyCallOp::Adaptor adaptor,
                              ::mlir::ConversionPatternRewriter &rewriter) const override;
    };

    struct FromBinOp : public ::mlir::OpConversionPattern<::plier::BinOp>
    {
        using OpConversionPattern::OpConversionPattern;

        ::mlir::LogicalResult
              matchAndRewrite(::plier::BinOp op,
                              ::plier::BinOp::Adaptor adaptor,
                              ::mlir::ConversionPatternRewriter &rewriter) const override;
    };

    // Lowering arange to LinAlg
    struct ARangeLowering : public ::mlir::OpConversionPattern<::ptensor::ARangeOp>
    {
        using OpConversionPattern::OpConversionPattern;

        ::mlir::LogicalResult
              matchAndRewrite(::ptensor::ARangeOp op,
                              ::ptensor::ARangeOp::Adaptor adaptor,
                              ::mlir::ConversionPatternRewriter &rewriter) const override;
    };

    // Lowering element-wise binary operations to LinAlg
    struct EWBinOpLowering : public ::mlir::OpConversionPattern<::ptensor::EWBinOp>
    {
        using OpConversionPattern::OpConversionPattern;

        ::mlir::LogicalResult
              matchAndRewrite(::ptensor::EWBinOp op,
                              ::ptensor::EWBinOp::Adaptor adaptor,
                              ::mlir::ConversionPatternRewriter &rewriter) const override;

        static ::mlir::Type getEWBinOpRType(::mlir::ConversionPatternRewriter &rewriter,
                                            EWBinOpId op,
                                            const ::mlir::Type & lhsType,
                                            const ::mlir::Type & rhsType);
    };

} // namespace ptensor
