#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

namespace ccw {

/// ImplicitLocOpBuilder maintains a 'current location', allowing use of the
/// create<> method without specifying the location.  It is otherwise the same
/// as OpBuilder.
class ImplicitLocOpBuilder : public mlir::OpBuilder {
public:
  /// OpBuilder has a bunch of convenience constructors - we support them all
  /// with the additional Location.
  template <typename... T>
  ImplicitLocOpBuilder(mlir::Location loc, T &&...operands)
      : mlir::OpBuilder(std::forward<T>(operands)...), curLoc(loc) {}

  /// Create a builder and set the insertion point to before the first operation
  /// in the block but still inside the block.
  static ImplicitLocOpBuilder atBlockBegin(mlir::Location loc, mlir::Block *block,
                                           Listener *listener = nullptr) {
    return ImplicitLocOpBuilder(loc, block, block->begin(), listener);
  }

  /// Create a builder and set the insertion point to after the last operation
  /// in the block but still inside the block.
  static ImplicitLocOpBuilder atBlockEnd(mlir::Location loc, mlir::Block *block,
                                         Listener *listener = nullptr) {
    return ImplicitLocOpBuilder(loc, block, block->end(), listener);
  }

  /// Create a builder and set the insertion point to before the block
  /// terminator.
  static ImplicitLocOpBuilder atBlockTerminator(mlir::Location loc, mlir::Block *block,
                                                Listener *listener = nullptr) {
    auto *terminator = block->getTerminator();
    assert(terminator != nullptr && "the block has no terminator");
    return ImplicitLocOpBuilder(loc, block, mlir::Block::iterator(terminator),
                                listener);
  }

  /// Accessors for the implied location.
  mlir::Location getLoc() const { return curLoc; }
  void setLoc(mlir::Location loc) { curLoc = loc; }

  // We allow clients to use the explicit-loc version of create as well.
  using mlir::OpBuilder::create;
  using mlir::OpBuilder::createOrFold;

  /// Create an operation of specific op type at the current insertion point and
  /// location.
  template <typename OpTy, typename... Args>
  OpTy create(Args &&...args) {
    return create<OpTy>(curLoc, std::forward<Args>(args)...);
    //return OpTy::create(*this, curLoc, std::forward<Args>(args)...);
  }

  /// Create an operation of specific op type at the current insertion point,
  /// and immediately try to fold it. This functions populates 'results' with
  /// the results after folding the operation.
  template <typename OpTy, typename... Args>
  void createOrFold(llvm::SmallVectorImpl<mlir::Value> &results, Args &&...args) {
    mlir::OpBuilder::createOrFold<OpTy>(results, curLoc, std::forward<Args>(args)...);
  }

  /// Overload to create or fold a single result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<mlir::OpTrait::OneResult>(), mlir::Value>
  createOrFold(Args &&...args) {
    return mlir::OpBuilder::createOrFold<OpTy>(curLoc, std::forward<Args>(args)...);
  }

  /// Overload to create or fold a zero result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<mlir::OpTrait::ZeroResults>(), OpTy>
  createOrFold(Args &&...args) {
    return OpBuilder::createOrFold<OpTy>(curLoc, std::forward<Args>(args)...);
  }

  /// This builder can also be used to emit diagnostics to the current location.
  mlir::InFlightDiagnostic
  emitError(const llvm::Twine &message = llvm::Twine()) {
    return mlir::emitError(curLoc, message);
  }
  mlir::InFlightDiagnostic
  emitWarning(const llvm::Twine &message = llvm::Twine()) {
    return mlir::emitWarning(curLoc, message);
  }
  mlir::InFlightDiagnostic
  emitRemark(const llvm::Twine &message = llvm::Twine()) {
    return mlir::emitRemark(curLoc, message);
  }

private:
  mlir::Location curLoc;
};


}

namespace mlir::triton::instrument {

constexpr int numMemTypes = getMaxEnumValForMemType() + 1;

constexpr int NUM_THREADS = 16;
constexpr int TMA_THREAD_OFFSET = NUM_THREADS;
constexpr int TC_THREAD_OFFSET = TMA_THREAD_OFFSET + NUM_THREADS;
constexpr int TOTAL_NUM_THREADS = TC_THREAD_OFFSET + NUM_THREADS;
constexpr int THREADS_BITMASK_SIZE = llvm::NextPowerOf2(TOTAL_NUM_THREADS);

Operation *createStoreScratchMemory(OpBuilder &b, Location loc, Value alloc,
                                    Value tensor, RankedTensorType tensorType);
Operation *createLoadScratchMemory(OpBuilder &b, Location loc, Value alloc,
                                   RankedTensorType tensorType);
Value expandOuterSlicedDim(OpBuilder &b, Location loc, Value tensor);
TypedValue<RankedTensorType> createConstIntTensor(OpBuilder &builder,
                                                  Location loc, int64_t val,
                                                  RankedTensorType tensorType,
                                                  bool isSigned = false);
FuncOp getEntryPoint(ModuleOp module);
gpu::DistributedEncodingTrait
getSingleDimSliceEncoding(gpu::BlockedEncodingAttr encoding, int dim);

// Map from IR region to ConSan auxiliary data. Auxiliary data is a value
// and an optional type, for values that are stored in the scratch memory.
struct AuxDataMap {
  struct RegionToValueMap {
    struct ValueType {
      Value value;
      Type type = nullptr;
    };
    DenseMap<Region *, ValueType> values;
    ValueType &operator[](Region *region) { return values[region]; }
    ValueType &operator[](Operation *op) {
      return values[getEnclosingParitionOrFunctionRegion(op)];
    }
    bool empty() const { return values.empty(); }

  private:
    Region *getEnclosingParitionOrFunctionRegion(Operation *op);
  };

  // Please see TritonInstrumentOps.td for more information on the auxiliary
  // data structures.
  RegionToValueMap buffers[numMemTypes];
  RegionToValueMap barriers;
  RegionToValueMap barrierStates;

  RegionToValueMap writeVisibility[numMemTypes];
  RegionToValueMap writeTracking[numMemTypes];
  RegionToValueMap readVisibility[numMemTypes];
  RegionToValueMap readTracking[numMemTypes];
  RegionToValueMap asyncCpCommits;
  RegionToValueMap wgmmaCommits;
  RegionToValueMap lock;
  RegionToValueMap waiting;

  void populateAndPassToWarpSpecialize(ModuleOp module);

private:
  void getBuffersAndBarriers(ModuleOp module,
                             SmallVector<SmallVector<int32_t>, 2> &bufValues,
                             SmallVector<int32_t> &barrierValues);
  void passToWarpSpecialize(triton::FuncOp func,
                            AuxDataMap::RegionToValueMap::ValueType value,
                            RegionToValueMap &map);
  void createInWarpSpecialize(
      triton::FuncOp func, RegionToValueMap &map,
      std::function<RegionToValueMap::ValueType(ccw::ImplicitLocOpBuilder &)>
          createFn);
};

} // namespace mlir::triton::instrument
