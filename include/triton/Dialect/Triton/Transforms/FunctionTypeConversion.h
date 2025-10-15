#ifndef TRITON_DIALECT_TRITON_TRANSFORMS_FUNCTION_TYPE_CONVERSION_H_
#define TRITON_DIALECT_TRITON_TRANSFORMS_FUNCTION_TYPE_CONVERSION_H_
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton {

/**
 * @brief Provides helper patterns for converting triton function operations
 * using a type converter.
 *
 * Note we cannot use upstream passes for this because they are unaware of
 * tt.call and tt.return.
 */
void populateFunctionTypeConversions(const TypeConverter &converter,
                                     RewritePatternSet &patterns);

void replaceOpWithMultiple(PatternRewriter &rewriter,
    Operation *op, SmallVector<SmallVector<Value>> &&newValues);

template <typename RangeT = ValueRange>
void replaceOpWithMultiple(PatternRewriter &rewriter, Operation *op, ArrayRef<RangeT> newValues) {
  replaceOpWithMultiple(rewriter, op,
                        llvm::to_vector_of<SmallVector<Value>>(newValues));
}
template <typename RangeT>
void replaceOpWithMultiple(PatternRewriter &rewriter, Operation *op, RangeT &&newValues) {
  replaceOpWithMultiple(rewriter, op,
                        ArrayRef(llvm::to_vector_of<ValueRange>(newValues)));
}

} // namespace mlir::triton

#endif // TRITON_DIALECT_TRITON_TRANSFORMS_FUNCTION_TYPE_CONVERSION_H_
