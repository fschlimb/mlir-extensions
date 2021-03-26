#pragma once

#include <mlir/Support/LogicalResult.h>
#include <mlir/IR/PatternMatch.h>

namespace plier
{
mlir::LogicalResult applyCSE(mlir::Region& region, mlir::PatternRewriter& rewriter, bool recursive);
mlir::LogicalResult applyCSE(mlir::Region& region, bool recursive);

template<typename Op, bool Recursive>
struct CSERewrite : public mlir::OpRewritePattern<Op>
{
    CSERewrite(mlir::MLIRContext *context):
        mlir::OpRewritePattern<Op>(context, /*benefit*/1) {} // TODO: benefit=0

    mlir::LogicalResult matchAndRewrite(
        Op op, mlir::PatternRewriter &rewriter) const override
    {
        return ::plier::applyCSE(op.getRegion(), rewriter, Recursive);
    }
};
}