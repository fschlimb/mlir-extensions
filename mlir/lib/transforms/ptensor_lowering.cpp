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

#include "mlir-extensions/transforms/ptensor_lowering.hpp"
#include "mlir-extensions/dialect/plier/dialect.hpp"
#include "mlir-extensions/dialect/plier_util/dialect.hpp"

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>

static ::mlir::Type makeSignlessType(::mlir::Type type)
{
    if (auto shaped = type.dyn_cast<mlir::ShapedType>()) {
        auto origElemType = shaped.getElementType();
        auto signlessElemType = makeSignlessType(origElemType);
        return shaped.clone(signlessElemType);
    } else if (auto intType = type.dyn_cast<::mlir::IntegerType>()) {
        if (!intType.isSignless())
            return ::mlir::IntegerType::get(intType.getContext(), intType.getWidth());
    }
    return type;
}

enum class ArrayLayout { C, F, A };

static bool parseLayout(llvm::StringRef &name, ArrayLayout &layout) {
  if (name.consume_back("C")) {
    layout = ArrayLayout::C;
    return true;
  }
  if (name.consume_back("F")) {
    layout = ArrayLayout::F;
    return true;
  }
  if (name.consume_back("A")) {
    layout = ArrayLayout::A;
    return true;
  }
  return false;
}

template <typename T>
static bool consumeIntBack(llvm::StringRef &name, T &result) {
  unsigned len = 0;
  auto tmp_name = name;
  while (!tmp_name.empty() && std::isdigit(tmp_name.back())) {
    ++len;
    tmp_name = tmp_name.drop_back();
  }
  tmp_name = name.substr(name.size() - len);
  if (!tmp_name.consumeInteger<T>(10, result)) {
    name = name.substr(0, name.size() - len);
    return true;
  }
  return false;
}

struct ArrayDesc {
  unsigned dims = 0;
  ArrayLayout layout = {};
  llvm::StringRef name;
};

static llvm::Optional<ArrayDesc> parseArrayDesc(llvm::StringRef &name) {
  unsigned num_dims = 0;
  ArrayLayout layout = {};
  if (name.consume_front("array(") && name.consume_back(")") &&
      parseLayout(name, layout) && name.consume_back(", ") &&
      name.consume_back("d") && consumeIntBack(name, num_dims) &&
      name.consume_back(", ") && !name.empty()) {
    return ArrayDesc{num_dims, layout, name};
  }
  return {};
}

// Convert a scalar type (string) from Plier to MLIR type
// return ::mlir::NoneType if unsupported
static mlir::Type getScalarType(mlir::Builder & b, const llvm::StringRef & name)
{
    if(name == "bool") return b.getIntegerType(1);
    if(name == "int8") return b.getIntegerType(8); //, true);
    if(name == "int16") return b.getIntegerType(16); //, true);
    if(name == "int32") return b.getIntegerType(32); //, true);
    if(name == "int64") return b.getIntegerType(64); //, true);
    
    if(name == "uint8") return b.getIntegerType(8, false);
    if(name == "uint16") return b.getIntegerType(16, false);
    if(name == "uint32") return b.getIntegerType(32, false);
    if(name == "uint64") return b.getIntegerType(64, false);
    
    // if(name == "int8_signless") return b.getIntegerType(8);
    // if(name == "int16_signless") return b.getIntegerType(16);
    // if(name == "int32_signless") return b.getIntegerType(32);
    // if(name == "int64_signless") return b.getIntegerType(64);
    // if(name == "index") return b.getIndexType();
    
    if(name == "float16") return b.getF16Type();
    if(name == "float32") return b.getF32Type();
    if(name == "float64") return b.getF64Type();

    return ::mlir::NoneType::get(b.getContext());;
}

// Convert any type (string) from Plier to MLIR type
// return ::mlir::NoneType if unsupported
static ::mlir::Type getType(mlir::Builder & b, const ::mlir::Type & t)
{
    auto typ = t.dyn_cast<plier::PyType>();
    auto nm = typ.getName();
    auto desc = parseArrayDesc(nm);
    if(desc) {
        auto styp = getScalarType(b, desc->name);
        llvm::SmallVector<int64_t> shp(desc->dims, ::mlir::ShapedType::kDynamicSize);
        return ::ptensor::PTensorType::get(b.getContext(), ::mlir::RankedTensorType::get(shp, styp), true);
    }
    return getScalarType(b, nm);
}

mlir::LogicalResult
ptensor::FromNumpyCall::matchAndRewrite(::plier::PyCallOp op,
                                        ::plier::PyCallOp::Adaptor adaptor,
                                        ::mlir::ConversionPatternRewriter &rewriter) const
{
    auto converter = *getTypeConverter();
    auto name = adaptor.func_name();
    //    auto rtyp = getType(rewriter, op.getType());
    auto rtyp = converter.convertType(op.getType());
    auto args = adaptor.args();

    if(name == "numpy.arange") {
        rewriter.replaceOpWithNewOp<::ptensor::ARangeOp>(op, rtyp, args[0], args[1], args[2], true);
        return ::mlir::success();
    }
    return ::mlir::failure();
};

mlir::LogicalResult
ptensor::FromBinOp::matchAndRewrite(::plier::BinOp op,
                                    ::plier::BinOp::Adaptor adaptor,
                                    ::mlir::ConversionPatternRewriter &rewriter) const
{
    auto converter = *getTypeConverter();
    auto lhs = adaptor.lhs();
    auto rhs = adaptor.rhs();

    auto lhstyp = lhs.getType().dyn_cast<::ptensor::PTensorType>();
    auto rhstyp = rhs.getType().dyn_cast<::ptensor::PTensorType>();
    if(lhstyp && rhstyp) {
        auto name = adaptor.op();
        auto rtyp = converter.convertType(op.getType());
        if(name == "+") {
            rewriter.replaceOpWithNewOp<::ptensor::EWBinOp>(op, rtyp, rewriter.getI32IntegerAttr(::ptensor::ADD), lhs, rhs);
            return ::mlir::success();
        } else if(name == "*") {
            rewriter.replaceOpWithNewOp<::ptensor::EWBinOp>(op, rtyp, rewriter.getI32IntegerAttr(::ptensor::MULTIPLY), lhs, rhs);
            return ::mlir::success();
        }
    }
    return ::mlir::failure();
};

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
        builder.create<mlir::linalg::YieldOp>(loc, ret.getResult());
    };

    auto x = rewriter.replaceOpWithNewOp<mlir::linalg::GenericOp>(op, ttyp, llvm::None, _tnsr.getResult(), maps, iterators, body);
    x.dump();
    std::cerr << "_Yey\n";
    return ::mlir::success();
}

template<typename OP, typename RTYPE>
struct EWBinOpAttr
{
    using opType = OP;
    using rType = RTYPE;
};

// Return a lowered op
mlir::Type ptensor::EWBinOpLowering::getEWBinOpRType(::mlir::ConversionPatternRewriter &rewriter,
                                                     ::ptensor::EWBinOpId op,
                                                     const ::mlir::Type & lhsType,
                                                     const ::mlir::Type & rhsType)
{
    switch(op) {
    case ADD:
    case ATAN2:
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
using BodyType = std::function<void(mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange args)>;
// array of body functions, for each binop one
//static std::array<BodyType*, ptensor::EWBINOPID_LAST> bodies;

// any body needs to close with a yield
template<typename T>
static void yield(mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::Type typ, T op)
{
    auto res = op.getResult();
    if(typ != res.getType()) {
        res = builder.create<plier::SignCastOp>(loc, typ, op).getResult();
    }
    builder.create<mlir::linalg::YieldOp>(loc, res);
}

template<typename IOP>
static BodyType buildTrivial(::mlir::Type typ)
{
    return [typ](mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange args) -> void {
        if(args[0].getType().isSignlessInteger()) {
            yield(builder, loc, typ, builder.create<IOP>(loc, args[0], args[1]));
        } else {
            assert("Only signless integers supported for binary ops" == nullptr);
        }
    };
}

// initialize array of builder functions
static BodyType getBodyBuilder(::ptensor::EWBinOpId bop, ::mlir::Type typ)
{
    switch(bop) {
    case ptensor::ADD:
        return buildTrivial<mlir::arith::AddIOp>(typ);
    // case ptensor::ATAN2] =
    case ptensor::FLOOR_DIVIDE:
        return buildTrivial<mlir::arith::FloorDivSIOp>(typ);
    // case ptensor::LOGADDEXP] =
    // case ptensor::LSHIFT] =
    // case ptensor::MATMUL] =
    case ptensor::MOD:
        return buildTrivial<mlir::arith::RemSIOp>(typ);
    case ptensor::MULTIPLY:
        return buildTrivial<mlir::arith::MulIOp>(typ);
    // case ptensor::POW] =
    case ptensor::SUBTRACT:
        return buildTrivial<mlir::arith::SubIOp>(typ);
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


mlir::LogicalResult
ptensor::EWBinOpLowering::matchAndRewrite(::ptensor::EWBinOp op,
                                          ::ptensor::EWBinOp::Adaptor adaptor,
                                          ::mlir::ConversionPatternRewriter &rewriter) const
{
    auto lhstyp = adaptor.lhs().getType().dyn_cast<::mlir::RankedTensorType>();
    auto rhstyp = adaptor.rhs().getType().dyn_cast<::mlir::RankedTensorType>();

    if(!lhstyp || !rhstyp) {
        return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto converter = *getTypeConverter();

    auto slcast = [&loc, &rewriter](auto o, auto ttyp) {
        auto etyp = ttyp.getElementType();
        if(etyp.isIntOrIndex() && !etyp.isSignlessInteger()) {
            auto slint = makeSignlessType(ttyp);
            return rewriter.create<::plier::SignCastOp>(loc, slint, o).getResult();
        }
        return o;
    };

    // Get operands into vec
    llvm::SmallVector<mlir::Value, 2> oprnds(2);
    oprnds[0] = slcast(adaptor.lhs(), lhstyp);
    oprnds[1] = slcast(adaptor.rhs(), rhstyp);

    // input tensors might have compatible but different types
    assert(oprnds[0].getType() == oprnds[1].getType());

    // the element type of a binop depends on the input arguments and the operation itself
    auto _t = converter.convertType(op.getType());
    auto shaped = _t.dyn_cast<::mlir::RankedTensorType>();
    assert(shaped);
    auto typ = shaped.getElementType();
    
    // build tensor using the resulting element type
    // the shape is not statically known, we need to retrieve it (it's the same as the input shapes)
    // FIXME shape broadcasting: input tensors might have compatible but different shapes
    auto rank = static_cast<unsigned>(shaped.getRank());
    llvm::SmallVector<mlir::Value> shp(rank);
    for(auto i : llvm::seq(0u, rank)) {
        shp[i] = rewriter.create<::mlir::tensor::DimOp>(loc, oprnds[0], i);
    }
    // create new tensor
    auto tnsr = rewriter.create<::mlir::linalg::InitTensorOp>(loc, shp, typ);

    // all maps are identity maps
    auto imap = ::mlir::AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    const ::mlir::AffineMap maps[] = {imap, imap, imap};
    // iterate in parallel
    llvm::SmallVector<mlir::StringRef> iterators(1, "parallel");

    // create binop as linalg::generic
    const auto bopid = (::ptensor::EWBinOpId)adaptor.op().cast<::mlir::IntegerAttr>().getInt();
    auto bodyBuilder = getBodyBuilder(bopid, typ);
    auto x = rewriter.replaceOpWithNewOp<::mlir::linalg::GenericOp>(op, tnsr.getType(), oprnds, tnsr.getResult(), maps, iterators, bodyBuilder).getResult(0);
    x.dump();
    return ::mlir::success();
}
