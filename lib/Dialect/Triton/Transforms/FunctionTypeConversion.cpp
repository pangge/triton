#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"

#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdlib>

namespace mlir::triton {

namespace {

SmallVector<Value> flattenValues(ArrayRef<ValueRange> values) {
  SmallVector<Value> ret;
  for (const auto &vs : values) {
    llvm::append_range(ret, vs);
  }
  return ret;
}

struct CallOpConversion : public OpConversionPattern<CallOp> {
  using OpConversionPattern<CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CallOp callOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<std::size_t> resultReplacementGrouping;
    llvm::SmallVector<Type> convertedResults;

    for (auto type : callOp->getResultTypes()) {
      const auto oldNumFlattenedResults = convertedResults.size();
      if (failed(getTypeConverter()->convertTypes(type, convertedResults))) {
        return failure();
      }
      resultReplacementGrouping.push_back(convertedResults.size() -
                                          oldNumFlattenedResults);
    }

    auto newCallOp = rewriter.create<CallOp>(
        callOp->getLoc(), callOp.getCallee(), convertedResults,
        flattenValues(adaptor.getOperands()));
    // Preserve any additional attributes that may have been set on the op
    newCallOp->setAttrs(callOp->getAttrs());

    SmallVector<ValueRange> replacements;
    std::size_t offset = 0;
    for (auto groupSize : resultReplacementGrouping) {
      replacements.push_back(newCallOp->getResults().slice(offset, groupSize));
      offset += groupSize;
    }

    //rewriter.replaceOpWithMultiple(callOp, replacements);
    replaceOpWithMultiple(rewriter, callOp, std::move(replacements));
    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<ReturnOp> {
  using OpConversionPattern<ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp returnOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newReturnOp = rewriter.create<ReturnOp>(
        returnOp->getLoc(), flattenValues(adaptor.getOperands()));
    // Preserve any additional attributes that may have been set on the op
    newReturnOp->setAttrs(returnOp->getAttrs());

    rewriter.replaceOp(returnOp, newReturnOp);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// FunctionOpInterfaceSignatureConversion
//===----------------------------------------------------------------------===//
// NOTE: Forked from mlir to support remapping argument attributes correctly in
// a one-to-many type conversion.

SmallVector<Attribute>
convertFuncOpAttrs(FunctionOpInterface funcOp,
                   TypeConverter::SignatureConversion &sigConv,
                   FunctionType newType) {
  if (newType.getNumInputs() == funcOp.getNumArguments()) {
    return {};
  }
  ArrayAttr allArgAttrs = funcOp.getAllArgAttrs();
  if (!allArgAttrs)
    return {};

  SmallVector<Attribute> newAttrs(newType.getNumInputs());
  for (auto i : llvm::seq(allArgAttrs.size())) {
    auto mapping = sigConv.getInputMapping(i);
    assert(mapping.has_value());
    auto outIdx = mapping->inputNo;
    newAttrs[outIdx] = allArgAttrs[i];
  }
  return newAttrs;
}

LogicalResult convertFuncOpTypes(FunctionOpInterface funcOp,
                                 const TypeConverter &typeConverter,
                                 ConversionPatternRewriter &rewriter) {
  FunctionType type = dyn_cast<FunctionType>(funcOp.getFunctionType());
  if (!type)
    return failure();

  // Convert the original function types.
  TypeConverter::SignatureConversion result(type.getNumInputs());
  SmallVector<Type, 1> newResults;
  if (failed(typeConverter.convertSignatureArgs(type.getInputs(), result)) ||
      failed(typeConverter.convertTypes(type.getResults(), newResults)) ||
      failed(rewriter.convertRegionTypes(&funcOp.getFunctionBody(),
                                         typeConverter, &result)))
    return failure();

  // Update the function signature in-place.
  auto newType = FunctionType::get(rewriter.getContext(),
                                   result.getConvertedTypes(), newResults);

  auto newArgAttrs = convertFuncOpAttrs(funcOp, result, newType);

  rewriter.modifyOpInPlace(funcOp, [&] {
    funcOp.setType(newType);
    if (!newArgAttrs.empty()) {
      funcOp.setAllArgAttrs(newArgAttrs);
    }
  });

  return success();
}

/// Create a default conversion pattern that rewrites the type signature of a
/// FunctionOpInterface op. This only supports ops which use FunctionType to
/// represent their type.
struct FunctionOpInterfaceSignatureConversion : public ConversionPattern {
  FunctionOpInterfaceSignatureConversion(StringRef functionLikeOpName,
                                         MLIRContext *ctx,
                                         const TypeConverter &converter,
                                         PatternBenefit benefit = 1)
      : ConversionPattern(converter, functionLikeOpName, benefit, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionOpInterface funcOp = cast<FunctionOpInterface>(op);
    return convertFuncOpTypes(funcOp, *typeConverter, rewriter);
  }
};

} // namespace

void populateFunctionTypeConversions(const TypeConverter &converter,
                                     RewritePatternSet &patterns) {
  auto context = patterns.getContext();
  patterns.add<FunctionOpInterfaceSignatureConversion>(
      triton::FuncOp::getOperationName(), context, converter);
  patterns.add<CallOpConversion, ReturnOpConversion>(converter, context);
}

void replaceOpWithMultiple(PatternRewriter &rewriter,
    Operation *op, SmallVector<SmallVector<Value>> &&newValues) {
  assert(op->getNumResults() == newValues.size() &&
         "incorrect # of replacement values");
  // If the current insertion point is before the erased operation, we adjust
  // the insertion point to be after the operation.
  if (rewriter.getInsertionPoint() == op->getIterator())
    rewriter.setInsertionPointAfter(op);

  SmallVector<Value> values;
  for (auto &vec : newValues) {
    values.append(vec.begin(), vec.end());
  }
  rewriter.replaceOp(op, ValueRange(values));
}

} // namespace mlir::triton
