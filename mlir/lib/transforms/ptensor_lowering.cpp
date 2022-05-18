// Copyright 2022 Intel Corporation
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

#include "mlir-extensions/transforms/ptensor_lowering.hpp"
#include "mlir-extensions/dialect/plier/dialect.hpp"
#include "mlir-extensions/dialect/plier_util/dialect.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>

mlir::LogicalResult
ptensor::ARangeLowering::matchAndRewrite(ptensor::ARangeOp op, mlir::PatternRewriter &rewriter) const
{
    // Get Operands
    auto loc = op.getLoc();
    auto start = op.start();
    auto stop = op.stop();
    auto step = op.step();

    auto ityp = rewriter.getI64Type();
    if (start.getType() != ityp) {
        start = rewriter.create<plier::SignCastOp>(loc, ityp, start);
    }
    if (stop.getType() != ityp) {
        stop = rewriter.create<plier::SignCastOp>(loc, ityp, stop);
    }
    if (step.getType() != ityp) {
        step = rewriter.create<plier::SignCastOp>(loc, ityp, step);
    }

    // Create constants 0, 1, -1 for later
    auto zattr = rewriter.getI64IntegerAttr(0);
    auto zero = rewriter.create<mlir::arith::ConstantOp>(loc, zattr).getResult();
    auto oattr = rewriter.getI64IntegerAttr(1);
    auto one = rewriter.create<mlir::arith::ConstantOp>(loc, oattr).getResult();
    auto mattr = rewriter.getI64IntegerAttr(-1);
    auto mone = rewriter.create<mlir::arith::ConstantOp>(loc, mattr).getResult();

    // Compute number of elements as (stop - start + step + (step < 0 ? 1 : -1)) / step
    auto cnd = rewriter.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ult, step, zero);
    auto inc = rewriter.create<mlir::arith::SelectOp>(loc, cnd, one, mone);
    auto tmp1 = rewriter.create<mlir::arith::AddIOp>(loc, stop, step);
    auto tmp2 = rewriter.create<mlir::arith::AddIOp>(loc, tmp1, inc);
    auto tmp3 = rewriter.create<mlir::arith::SubIOp>(loc, tmp2, start);
    auto cnt = rewriter.create<mlir::arith::DivUIOp>(loc, tmp3, step).getResult();
    cnt = rewriter.create<plier::SignCastOp>(loc, mlir::IndexType::get(cnt.getType().getContext()), cnt);

    // create a 1d tensor of size cnt
    auto typ = ityp; // rewriter.getIntegerType(64, true);
    llvm::SmallVector<mlir::Value> shp(1);
    shp[0] = cnt;
    auto _tnsr = rewriter.create<mlir::linalg::InitTensorOp>(loc, shp, typ);

    // fill with arange values
    // map needed for output only (we have no input tensor)
    const mlir::AffineMap maps[] = {
        mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext())
    };
    llvm::SmallVector<mlir::StringRef> iterators(1, "parallel");

    // The body; accepting no input, the lambda simply captures start and step
    auto body = [&start, &step, &typ, &ityp](mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args) {
        auto dim = builder.getI64IntegerAttr(0);
        auto idx = builder.create<mlir::linalg::IndexOp>(loc, dim);
        auto _idx = builder.create<mlir::arith::IndexCastOp>(loc, ityp, idx);
        auto tmp = builder.create<mlir::arith::MulIOp>(loc, step, _idx);
        auto val = builder.create<mlir::arith::AddIOp>(loc, start, tmp);
        // auto _val = builder.create<mlir::arith::SIToFPOp>(loc, typ, val);
        builder.create<mlir::linalg::YieldOp>(loc, val.getResult());
    };

    auto rtyp = mlir::RankedTensorType::get({-1}, typ);
    rewriter.replaceOpWithNewOp<mlir::linalg::GenericOp>(op, rtyp, llvm::None, _tnsr.getResult(), maps, iterators, body);
    return mlir::success();
#if 0
    auto arange = rewriter.create<mlir::linalg::GenericOp>(loc, tnsr.getType(), llvm::None, tnsr, maps, iterators, body).getResult(0);

    // finally cast to memref
    auto tensorType = arange.getType().dyn_cast<mlir::RankedTensorType>();
    auto memrefType = mlir::MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    rewriter.replaceOpWithNewOp<::mlir::bufferization::ToMemrefOp>(op, memrefType, arange);

    return mlir::success();
#endif // 0
}

mlir::LogicalResult
ptensor::EWBinOpLowering::matchAndRewrite(ptensor::EWBinOp op, mlir::PatternRewriter &rewriter) const
{
    auto loc = op.getLoc();

    // Get operands into vec
    llvm::SmallVector<mlir::Value, 2> oprnds(2);
    oprnds[0] = op.lhs();
    oprnds[1] = op.rhs();
    
    // result has same type and shape as oprnds
    auto rtt = op.lhs().getType().dyn_cast<mlir::RankedTensorType>();
    assert(rtt);
    auto typ = rtt.getElementType();
    assert(rtt.hasRank());
    auto shaped = rtt.dyn_cast<mlir::ShapedType>();
    assert(shaped);
    auto rank = static_cast<unsigned>(shaped.getRank());
    llvm::SmallVector<mlir::Value> shp(rank);
    for(auto i : llvm::seq(0u, rank)) {
        shp[i] = rewriter.create<mlir::tensor::DimOp>(loc, oprnds[0], i);
    }
    // create new tensor
    auto tnsr = rewriter.create<mlir::linalg::InitTensorOp>(loc, shp, typ);

    // all maps are identiy maps
    auto imap = mlir::AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    const mlir::AffineMap maps[] = {imap, imap, imap};
    // iterate in parallel
    llvm::SmallVector<mlir::StringRef> iterators(1, "parallel");

    // The body; accepting 2 inputs
    auto body = [](mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args) {
        auto val = builder.create<mlir::arith::AddIOp>(loc, args[0], args[1]);
        builder.create<mlir::linalg::YieldOp>(loc, val.getResult());
    };

    // create binop as linalg::generic
    auto rtyp = mlir::RankedTensorType::get({-1}, typ);
    auto bo = rewriter.replaceOpWithNewOp<mlir::linalg::GenericOp>(op, tnsr.getType(), oprnds, tnsr.getResult(), maps, iterators, body).getResult(0);

#if 0
    // finally cast to memref
    auto bo = rewriter.create<mlir::linalg::GenericOp>(loc, tnsr.getType(), oprnds, tnsr.getResult(), maps, iterators, body).getResult(0);
    auto tensorType = bo.getType().dyn_cast<mlir::RankedTensorType>();
    auto memrefType = mlir::MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    rewriter.replaceOpWithNewOp<::mlir::bufferization::ToMemrefOp>(op, memrefType, bo);
#endif
    return mlir::success();
}
