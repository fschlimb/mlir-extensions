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

#include <iostream>

#include "mlir-extensions/Transforms/ptensor_lowering.hpp"
#include "mlir-extensions/Dialect/plier/dialect.hpp"
#include "mlir-extensions/Dialect/plier_util/dialect.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>


// return type without a sign
// copied from py_linalg_resolver.cpp
static mlir::Type makeSignlessType(mlir::Type type) {
  if (auto shaped = type.dyn_cast<mlir::ShapedType>()) {
    auto origElemType = shaped.getElementType();
    return makeSignlessType(origElemType);
  } else if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    if (!intType.isSignless())
      return mlir::IntegerType::get(intType.getContext(), intType.getWidth());
  }
  return type;
}

// creating operand cast to signless type if needed
// copied from py_linalg_resolver.cpp
static mlir::Value doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
                              mlir::Value val) {
  auto origType = val.getType();
  auto signlessType = makeSignlessType(origType);
  if (signlessType != origType)
    val = builder.createOrFold<plier::SignCastOp>(loc, signlessType, val);

  return val;
}

// creating operand cast to given type if needed
// copied from py_linalg_resolver.cpp
static mlir::Value doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
                              mlir::Value val, mlir::Type dstType) {
  auto origType = val.getType();
  if (dstType != origType)
    val = builder.createOrFold<plier::SignCastOp>(loc, dstType, val);

  return val;
}


// convert numpy calls and their return types from Plier to PTensor
// as we use a type converter, operands are provided as converted types in adaptor
mlir::LogicalResult
ptensor::FromNumpyCall::matchAndRewrite(::plier::PyCallOp op,
                                        ::plier::PyCallOp::Adaptor adaptor,
                                        ::mlir::ConversionPatternRewriter &rewriter) const
{
    auto converter = *getTypeConverter();
    auto name = adaptor.func_name();
    // convert return type
    auto rtyp = converter.convertType(op.getType());
    op.getType().dump();
    rtyp.dump();
    // get auto-converted args/operands
    auto args = adaptor.args();

    // currently we support arange and sum only
    if(name == "numpy.arange") {
        (void)rewriter.replaceOpWithNewOp<::ptensor::ARangeOp>(op, rtyp, args[0], args[1], args[2], true);
        return ::mlir::success();
    } else if(name == "numpy.sum") {
        if(!rtyp.isa<::ptensor::PTensorType>()) {
            rtyp = ::ptensor::PTensorType::get(rewriter.getContext(), ::mlir::RankedTensorType::get({}, rtyp), true);
        }
        (void)rewriter.replaceOpWithNewOp<::ptensor::ReductionOp>(op, rtyp, rewriter.getI32IntegerAttr(::ptensor::SUM), args[0]);
        return ::mlir::success();
    }
    return ::mlir::failure();
};

// convert binary operations their return types from Plier binary to PTensor
// as we use a type converter, operands are provided as converted types in adaptor
mlir::LogicalResult
ptensor::FromBinOp::matchAndRewrite(::plier::BinOp op,
                                    ::plier::BinOp::Adaptor adaptor,
                                    ::mlir::ConversionPatternRewriter &rewriter) const
{
    auto converter = *getTypeConverter();
    // get auto-converted operands
    auto lhs = adaptor.lhs();
    auto rhs = adaptor.rhs();

    auto lhstyp = lhs.getType().dyn_cast<::ptensor::PTensorType>();
    auto rhstyp = rhs.getType().dyn_cast<::ptensor::PTensorType>();
    // we expect PTensorTypes as operands
    if(lhstyp && rhstyp) {
        auto name = adaptor.op();
        auto rtyp = converter.convertType(op.getType());
        if(name == "+") {
            (void)rewriter.replaceOpWithNewOp<::ptensor::EWBinOp>(op, rtyp, rewriter.getI32IntegerAttr(::ptensor::ADD), lhs, rhs);
            return ::mlir::success();
        } else if(name == "*") {
            (void)rewriter.replaceOpWithNewOp<::ptensor::EWBinOp>(op, rtyp, rewriter.getI32IntegerAttr(::ptensor::MULTIPLY), lhs, rhs);
            return ::mlir::success();
        }
    }
    // fail if not PTensorType operands or unsupported op
    // will be retried if operands gets converted elsewhere
    return ::mlir::failure();
};

// convert PTensor's arange and its return type to Linalg/tensor
// we also need some arith and affine (for linalg::genericop)
mlir::LogicalResult
ptensor::ARangeLowering::matchAndRewrite(::ptensor::ARangeOp op,
                                         ::ptensor::ARangeOp::Adaptor adaptor,
                                         ::mlir::ConversionPatternRewriter &rewriter) const
{
    auto loc = op.getLoc();
    auto converter = *getTypeConverter();

    // Get Operands
    auto start = adaptor.start();
    auto stop = adaptor.stop();
    auto step = adaptor.step();

    // we operator on signless integers
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
    auto cnd = rewriter.create<mlir::arith::CmpIOp>(loc, ::mlir::arith::CmpIPredicate::ult, step, zero);
    auto inc = rewriter.create<mlir::arith::SelectOp>(loc, cnd, one, mone);
    auto tmp1 = rewriter.create<mlir::arith::AddIOp>(loc, stop, step);
    auto tmp2 = rewriter.create<mlir::arith::AddIOp>(loc, tmp1, inc);
    auto tmp3 = rewriter.create<mlir::arith::SubIOp>(loc, tmp2, start);
    auto cnt = rewriter.create<mlir::arith::DivUIOp>(loc, tmp3, step).getResult();
    cnt = rewriter.create<plier::SignCastOp>(loc, ::mlir::IndexType::get(cnt.getType().getContext()), cnt);

    // create a 1d tensor of size cnt
    auto ttyp = converter.convertType(op.getType()).dyn_cast<::mlir::RankedTensorType>();
    assert(ttyp);
    auto typ = ttyp.getElementType();
    llvm::SmallVector<mlir::Value> shp(1);
    shp[0] = cnt;
    auto _tnsr = rewriter.create<mlir::linalg::InitTensorOp>(loc, shp, typ);

    // fill with arange values
    // map needed for output only (we have no input tensor)
    const ::mlir::AffineMap maps[] = {
        ::mlir::AffineMap::getMultiDimIdentityMap(1, rewriter.getContext())
    };
    llvm::SmallVector<mlir::StringRef> iterators(1, "parallel");

    // The body; accepting no input, the lambda simply captures start and step
    auto body = [&start, &step, &typ, &ityp](mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange args) {
        auto dim = builder.getI64IntegerAttr(0);
        auto idx = builder.create<mlir::linalg::IndexOp>(loc, dim);
        auto _idx = builder.create<mlir::arith::IndexCastOp>(loc, ityp, idx);
        auto tmp = builder.create<mlir::arith::MulIOp>(loc, step, _idx);
        auto val = builder.create<mlir::arith::AddIOp>(loc, start, tmp);
        auto ret = builder.create<plier::SignCastOp>(loc, typ, val);
        // auto _val = builder.create<mlir::arith::SIToFPOp>(loc, typ, val);
        (void)builder.create<mlir::linalg::YieldOp>(loc, ret.getResult());
    };

    (void)rewriter.replaceOpWithNewOp<mlir::linalg::GenericOp>(op, ttyp, llvm::None, _tnsr.getResult(), maps, iterators, body);
    return ::mlir::success();
}




// function type for building body for linalg::generic
using BodyType = std::function<void(mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange args)>;

// any genericOp body needs to close with a yield
// we also add a cast op to "typ" if needed
template<typename T>
static void yield(mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Type typ, T op)
{
    auto res = op.getResult();
    if(typ != res.getType()) {
        res = builder.create<plier::SignCastOp>(loc, typ, op).getResult();
    }
    (void)builder.create<mlir::linalg::YieldOp>(loc, res);
}

// trivial builders have simple arith equivalents
// the arith ops are template arguments, one for ints and one for floats
// currently only integers and floats are supported
// currently unsigned int ops are not supported
template<typename IOP, typename FOP = IOP>
static BodyType buildTrivial(::mlir::Type typ)
{
    return [typ](mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange args) -> void {
        auto lhs = doSignCast(builder, loc, args[0]);
        auto rhs = doSignCast(builder, loc, args[1]);
        if(lhs.getType().isIntOrIndex()) {
            yield(builder, loc, typ, builder.create<IOP>(loc, lhs, rhs));
        } else if(lhs.getType().isIntOrIndexOrFloat()) {
            yield(builder, loc, typ, builder.create<FOP>(loc, lhs, rhs));
        } else {
            assert("Only integers and floats supported for binary ops" == nullptr);
        }
    };
}

// get a body builder for given binary operation and result type
// we accept a result type to insert a cast after the operation if needed
static BodyType getBodyBuilder(::ptensor::EWBinOpId bop, ::mlir::Type typ)
{
    switch(bop) {
    case ptensor::ADD:
        return buildTrivial<mlir::arith::AddIOp, mlir::arith::AddFOp>(typ);
    // case ptensor::ATAN2] =
    case ptensor::FLOOR_DIVIDE:
        return buildTrivial<mlir::arith::FloorDivSIOp>(typ);
    // case ptensor::LOGADDEXP] =
    // case ptensor::LSHIFT] =
    // case ptensor::MATMUL] =
    case ptensor::MAXIMUM:
        return buildTrivial<mlir::arith::MaxSIOp, mlir::arith::MaxFOp>(typ);
    case ptensor::MINIMUM:
        return buildTrivial<mlir::arith::MinSIOp, mlir::arith::MinFOp>(typ);
    case ptensor::MODULO:
        return buildTrivial<mlir::arith::RemSIOp, mlir::arith::RemFOp>(typ);
    case ptensor::MULTIPLY:
        return buildTrivial<mlir::arith::MulIOp, mlir::arith::MulFOp>(typ);
    // case ptensor::POW] =
    case ptensor::SUBTRACT:
        return buildTrivial<mlir::arith::SubIOp, mlir::arith::SubFOp>(typ);
    // case ptensor::TRUE_DIVIDE] =
    // case ptensor::BITWISE_AND] =
    // case ptensor::BITWISE_LEFT_SHIFT] =
    // case ptensor::BITWISE_OR] =
    // case ptensor::BITWISE_RIGHT_SHIFT] =
    // case ptensor::BITWISE_XOR] =

    // case ptensor::EQUAL] =
    // case ptensor::GREATER] =
    // case ptensor::GREATER_EQUAL] =
    // case ptensor::LESS] =
    // case ptensor::LESS_EQUAL] =
    // case ptensor::LOGICAL_AND] =
    // case ptensor::LOGICAL_OR] =
    // case ptensor::LOGICAL_XOR] =
    // case ptensor::NOT_EQUAL] =
    default:
        assert("unsupported elementwise binary operation" == nullptr);
    };
}


// convert PTensor's elementwise binary operations and their return type to Linalg/tensor
// the given op's type is expected to convert to the apprioprate type (shape and element-type)
// we also need some arith and affine (for linalg::genericop)
mlir::LogicalResult
ptensor::EWBinOpLowering::matchAndRewrite(::ptensor::EWBinOp op,
                                          ::ptensor::EWBinOp::Adaptor adaptor,
                                          ::mlir::ConversionPatternRewriter &rewriter) const
{
    // we expect RankedTensorType as operands
    auto lhstyp = adaptor.lhs().getType().dyn_cast<::mlir::RankedTensorType>();
    auto rhstyp = adaptor.rhs().getType().dyn_cast<::mlir::RankedTensorType>();
    if(!lhstyp || !rhstyp) {
        // fail if not, will be retired if operands get converted elsewhere
        return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto converter = *getTypeConverter();

    // Get signless operands into vec
    llvm::SmallVector<mlir::Value, 2> oprnds = {adaptor.lhs(), adaptor.rhs()};

    // input tensors might have compatible but different types
    assert(oprnds[0].getType() == oprnds[1].getType());

    // the element type of a binop depends on the input arguments and the operation itself
    // we assume this had beeen taken care of and simply use the op's converted type
    auto _t = converter.convertType(op.getType());
    auto shaped = _t.dyn_cast<::mlir::RankedTensorType>();
    assert(shaped);
    auto typ = shaped.getElementType();
    
    // build tensor using the resulting element type
    // the shape is not statically known, we need to retrieve it (it's the same as the input shapes)
    // FIXME shape broadcasting: input tensors might have compatible but different shapes
    auto rank = static_cast<unsigned>(shaped.getRank());
    llvm::SmallVector<mlir::Value> shp(rank);
    llvm::SmallVector<mlir::StringRef> iterators(rank);
    for(auto i : llvm::seq(0u, rank)) {
        shp[i] = rewriter.create<::mlir::tensor::DimOp>(loc, oprnds[0], i);
        // iterate in parallel
        iterators[i] = "parallel";
    }
    // create new tensor
    auto tnsr = rewriter.create<::mlir::linalg::InitTensorOp>(loc, shp, typ);

    // all maps are identity maps
    auto imap = ::mlir::AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    const ::mlir::AffineMap maps[] = {imap, imap, imap};

    // create binop as linalg::generic
    const ::ptensor::EWBinOpId bopid = (::ptensor::EWBinOpId)adaptor.op().cast<::mlir::IntegerAttr>().getInt();
    auto bodyBuilder = getBodyBuilder(bopid, typ);
    (void)rewriter.replaceOpWithNewOp<::mlir::linalg::GenericOp>(op, tnsr.getType(), oprnds, tnsr.getResult(), maps, iterators, bodyBuilder).getResult(0);
    return ::mlir::success();
}

// get a body builder for giben binary operation and result type
// we accept a result type to insert a cast after the operation if needed
static BodyType getBodyBuilder(::ptensor::ReduceOpId rop, ::mlir::Type typ)
{
    switch(rop) {
    case ::ptensor::PROD:
        return getBodyBuilder(::ptensor::MULTIPLY, typ);
    case ::ptensor::SUM:
        return getBodyBuilder(::ptensor::ADD, typ);
    case ::ptensor::MAX:
        return getBodyBuilder(::ptensor::MAXIMUM, typ);
    case ::ptensor::MIN:
        return getBodyBuilder(::ptensor::MINIMUM, typ);
    case ::ptensor::MEAN:
    case ::ptensor::STD:
    case ::ptensor::VAR:
    default:
        assert("unsupported reduction operation" == nullptr);
    };
}

// convert PTensor's reduction operations and their return type to Linalg/tensor
// the given op's type is expected to convert to the apprioprate type (shape and element-type)
// we also need some arith and affine (for linalg::genericop)
// FIXME reduction over a subset of dimensions
::mlir::LogicalResult
ptensor::ReductionOpLowering::matchAndRewrite(::ptensor::ReductionOp op,
                                              ::ptensor::ReductionOp::Adaptor adaptor,
                                              ::mlir::ConversionPatternRewriter &rewriter) const
{
    // we expect RankedTensorType as operands
    auto inptyp = adaptor.input().getType().dyn_cast<::mlir::RankedTensorType>();
    if(!inptyp) {
        // fail if not, will be retired if operands get converted elsewhere
        return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto converter = *getTypeConverter();

    // Get signless operands into vec
    llvm::SmallVector<mlir::Value, 1> oprnds = {adaptor.input()};

    // determine resulting element type from converted op-type
    auto _t = converter.convertType(op.getType());
    auto shaped = _t.dyn_cast<::mlir::RankedTensorType>();
    assert(shaped);
    auto typ = shaped.getElementType();
    auto sltyp = makeSignlessType(typ);

    // build tensor using the resulting element type and shape
    // FIXME support reduction dimensions
    auto rank = static_cast<unsigned>(shaped.getRank());
    assert(rank==0);
    llvm::SmallVector<mlir::Value> shp(0); //::mlir::ShapedType::kDynamicSize;
    // create new tensor
    auto zattr = rewriter.getI64IntegerAttr(0);
    auto zero = rewriter.create<mlir::arith::ConstantOp>(loc, zattr).getResult();
    auto _tnsr = rewriter.create<::mlir::linalg::InitTensorOp>(loc, shp, sltyp).getResult();
    auto tnsr = rewriter.create<::mlir::linalg::FillOp>(loc, zero, _tnsr);

    // rank/num-dims of input
    auto irank = static_cast<unsigned>(inptyp.getRank());
    // input maps are identity maps
    auto imap = ::mlir::AffineMap::getMultiDimIdentityMap(irank, rewriter.getContext());
    // output map is "*->()"
    auto omap = ::mlir::AffineMap::get(irank, 0, rewriter.getContext());
    const ::mlir::AffineMap maps[] = {imap, omap};
    llvm::SmallVector<mlir::StringRef> iterators(irank, "reduction");

    // create reduction op as linalg::generic
    const ::ptensor::ReduceOpId ropid = (::ptensor::ReduceOpId)adaptor.op().cast<::mlir::IntegerAttr>().getInt();
    auto bodyBuilder = getBodyBuilder(ropid, sltyp);
    auto rtnsr = rewriter.create<::mlir::linalg::GenericOp>(loc, tnsr.getType(0), oprnds, tnsr.getResult(0), maps, iterators, bodyBuilder).getResult(0);
    auto rval = rewriter.create<::mlir::tensor::ExtractOp>(loc, sltyp, rtnsr, ::mlir::ValueRange());
    auto x = rewriter.replaceOpWithNewOp<::plier::SignCastOp>(op, typ, rval);
    x.dump();
    return ::mlir::success();
}
