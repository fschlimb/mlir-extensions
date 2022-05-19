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
}

template<typename OP, typename RTYPE>
struct EWBinOpAttr
{
    using opType = OP;
    using rType = RTYPE;
};

// Return a lowered op
mlir::Type ptensor::EWBinOpLowering::getEWBinOpRType(mlir::PatternRewriter &rewriter, ::ptensor::EWBinOpId op, const mlir::Type & lhsType, const mlir::Type & rhsType)
{
    switch(op) {
    case ADD:
    case ATAN2:
    case DIVIDE:
    case FLOOR_DIVIDE:
    case LOGADDEXP:
    case LSHIFT:
    case MATMUL:
    case MOD:
    case MULTIPLY:
    case POW:
    case SUBTRACT:
    case TRUE_DIVIDE:
    case BITWISE_AND:
    case BITWISE_LEFT_SHIFT:
    case BITWISE_OR:
    case BITWISE_RIGHT_SHIFT:
    case BITWISE_XOR: {
        auto rtt = lhsType.dyn_cast<mlir::RankedTensorType>();
        assert(rtt && rtt.getElementType().isIntOrIndex());
        return rtt.getElementType();
    }

    case EQUAL:
    case GREATER:
    case GREATER_EQUAL:
    case LESS:
    case LESS_EQUAL:
    case LOGICAL_AND:
    case LOGICAL_OR:
    case LOGICAL_XOR:
    case NOT_EQUAL:
        return rewriter.getI1Type();
    default:
        assert(nullptr == "unknown binary operation");
    };
}

// function type for building body for linalg::generic
using BodyType = void(mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args);
// array of body functions, for each binop one
static std::array<BodyType*, ptensor::EWBINOPID_LAST> bodies;

// any body needs to close with a yield
template<typename T>
static void yield(mlir::OpBuilder &builder, mlir::Location loc, T op)
{
    builder.create<mlir::linalg::YieldOp>(loc, op.getResult());
}

template<typename IOP>
static void buildTrivial(mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args)
{
    if(args[0].getType().isSignlessInteger()) {
        yield(builder, loc, builder.create<IOP>(loc, args[0], args[1]));
    }
}

// initialize array of builder functions
static void initBuilders()
{
    static bool inited = false;
    if(inited) return;

    // by default return failure
    bodies.fill([](mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange args) { assert("unsupported elementwise binary operation" == nullptr); });

    bodies[ptensor::ADD] = buildTrivial<mlir::arith::AddIOp>;
    // bodies[ptensor::ATAN2] =
    bodies[ptensor::FLOOR_DIVIDE] = buildTrivial<mlir::arith::FloorDivSIOp>;
    // bodies[ptensor::LOGADDEXP] =
    // bodies[ptensor::LSHIFT] =
    // bodies[ptensor::MATMUL] =
    bodies[ptensor::MOD] = buildTrivial<mlir::arith::RemSIOp>;
    bodies[ptensor::MULTIPLY] = buildTrivial<mlir::arith::MulIOp>;
    // bodies[ptensor::POW] =
    bodies[ptensor::SUBTRACT] = buildTrivial<mlir::arith::SubIOp>;
    // bodies[ptensor::TRUE_DIVIDE] =
    // bodies[ptensor::BITWISE_AND] =
    // bodies[ptensor::BITWISE_LEFT_SHIFT] =
    // bodies[ptensor::BITWISE_OR] =
    // bodies[ptensor::BITWISE_RIGHT_SHIFT] =
    // bodies[ptensor::BITWISE_XOR] =

    // bodies[ptensor::EQUAL] =
    // bodies[ptensor::GREATER] =
    // bodies[ptensor::GREATER_EQUAL] =
    // bodies[ptensor::LESS] =
    // bodies[ptensor::LESS_EQUAL] =
    // bodies[ptensor::LOGICAL_AND] =
    // bodies[ptensor::LOGICAL_OR] =
    // bodies[ptensor::LOGICAL_XOR] =
    // bodies[ptensor::NOT_EQUAL] =

    inited = true;
    return;
}


mlir::LogicalResult
ptensor::EWBinOpLowering::matchAndRewrite(::ptensor::EWBinOp op, mlir::PatternRewriter &rewriter) const
{
    initBuilders();

    const auto bopid = (::ptensor::EWBinOpId)op.op().cast<::mlir::IntegerAttr>().getInt();
    auto loc = op.getLoc();

    // Get operands into vec
    llvm::SmallVector<mlir::Value, 2> oprnds(2);
    oprnds[0] = op.lhs();
    oprnds[1] = op.rhs();

    // type coercion: input tensors might have compatible but different types
    assert(oprnds[0].getType() == oprnds[1].getType());

    // the element type of a binop depends on the input arguments and the operation itself
    auto typ = getEWBinOpRType(rewriter, bopid, oprnds[0].getType(), oprnds[1].getType());
    // build tensor using the resulting element type
    auto rtt = oprnds[0].getType().dyn_cast<mlir::RankedTensorType>();
    assert(rtt && rtt.hasRank());
    auto shaped = rtt.dyn_cast<mlir::ShapedType>();
    assert(shaped);
    // the shape is not statically known, we need to retrieve it (it's the same as the input shapes)
    // FIXME shape broadcasting: input tensors might have compatible but different shapes
    auto rank = static_cast<unsigned>(shaped.getRank());
    llvm::SmallVector<mlir::Value> shp(rank);
    for(auto i : llvm::seq(0u, rank)) {
        shp[i] = rewriter.create<mlir::tensor::DimOp>(loc, oprnds[0], i);
    }
    // create new tensor
    auto tnsr = rewriter.create<mlir::linalg::InitTensorOp>(loc, shp, typ);

    // all maps are identity maps
    auto imap = mlir::AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    const mlir::AffineMap maps[] = {imap, imap, imap};
    // iterate in parallel
    llvm::SmallVector<mlir::StringRef> iterators(1, "parallel");

    // create binop as linalg::generic
    auto rtyp = mlir::RankedTensorType::get({-1}, typ);
    rewriter.replaceOpWithNewOp<mlir::linalg::GenericOp>(op, tnsr.getType(), oprnds, tnsr.getResult(), maps, iterators, bodies[bopid]).getResult(0);

    return mlir::success();
}