#pragma once

#include <gtsam/nonlinear/CustomFactor.h>

#include <stdexcept>

#ifdef GTSAM_USE_TBB
#include <tbb/global_control.h>
#endif

namespace gtsam {

/**
 * MATLAB-only adapter for gtsam::CustomFactor.
 *
 * The core GTSAM CustomFactor stores a C++ callback. For MATLAB we cannot
 * store a raw mxArray* or function handle directly inside the factor because
 * the factor can outlive temporary MATLAB objects and can be copied by the
 * optimizer. Instead, the MATLAB layer registers the callback in a MATLAB-side
 * registry, passes only a numeric callback ID into C++, and this class calls
 * back into MATLAB through that registry when error() or linearize() requests
 * factor evaluation.
 */
class MatlabCustomFactor : public CustomFactor {
 private:
  /// Stable registry key owned by the MATLAB toolbox.
  size_t callback_id_;

  /// Opt-in snapshot restricted to declared keys; legacy callbacks see all values.
  bool factor_keys_only_;

#ifdef GTSAM_USE_TBB
  // MATLAB callback execution is not a good place to discover hidden
  // parallelism bugs. Serializing CustomFactor callback execution keeps the
  // MATLAB path predictable while debugging wrapper issues.
  std::shared_ptr<tbb::global_control> tbb_serial_guard_;
#endif

  static std::string registryFunctionName() {
    return "gtsam.customFactorRegistry";
  }

  static void destroyCallbackArgs(mxArray** args, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      if (args[i]) {
        mxDestroyArray(args[i]);
        args[i] = nullptr;
      }
    }
  }

  /// Create an owning snapshot, optionally restricted to declared keys.
  std::shared_ptr<Values> callbackValues(const Values& values) const {
    if (!factor_keys_only_) return std::make_shared<Values>(values);

    auto snapshot = std::make_shared<Values>();
    for (Key key : keys()) {
      // Values::insert clones the polymorphic Value, preserving its exact type.
      // Repeated keys occupy one Values entry; Jacobian order is unchanged.
      if (!snapshot->exists(key)) snapshot->insert(key, values.at(key));
    }
    return snapshot;
  }

  /// Check the storage contract required by the Eigen conversion helpers.
  static bool isDenseRealDouble(const mxArray* value) {
    return value && mxIsDouble(value) && !mxIsComplex(value) &&
           !mxIsSparse(value) && mxGetNumberOfDimensions(value) == 2;
  }

  /// Report a contract violation; the caller releases callback outputs.
  [[noreturn]] static void invalidOutput(const char* id, const char* message) {
    // Unwind optimizer-owned C++ temporaries before the generated wrapper
    // reports the error to MATLAB with an empty outer identifier.
    throw std::invalid_argument(std::string("[") + id + "] " + message);
  }

  /// Release a trapped MATLAB exception while preserving its diagnostics.
  [[noreturn]] static void throwCallbackError(mxArray* exception) {
    // The generated wrapper catches std::exception, including errors from
    // MATLAB throw/rethrow. Carry the original identifier and stack as text
    // through that existing boundary, after releasing callback-owned arrays.
    mxArray* args[4] = {exception, mxCreateString("extended"),
                       mxCreateString("hyperlinks"), mxCreateString("off")};
    mxArray* reports[2] = {nullptr, nullptr};
    reports[1] = mexCallMATLABWithTrap(1, reports, 4, args, "getReport");

    // mxGetProperty returns owning copies, unlike mxGetCell.
    mxArray* properties[2] = {nullptr, nullptr};
    std::string message;

    try {

      properties[0] = mxGetProperty(exception, 0, "identifier");
      if (reports[1]) properties[1] = mxGetProperty(exception, 0, "message");
      message = "CustomFactor callback [" +
                unwrap<std::string>(properties[0]) + "]:\n";
      message += unwrap<std::string>(reports[1] ? properties[1] : reports[0]);

    } catch (...) {
      destroyCallbackArgs(properties, 2);
      destroyCallbackArgs(reports, 2);
      destroyCallbackArgs(args, 4);
      throw;
    }

    destroyCallbackArgs(properties, 2);
    destroyCallbackArgs(reports, 2);
    destroyCallbackArgs(args, 4);
    throw std::runtime_error(message);
  }

  /// Validate output storage and dimensions before using Eigen conversions.
  void validateOutputs(const Values& values, mxArray* const* outputs,
                       bool jacobians) const {
    // Validate before conversion: unwrap assumes dense, real double storage.
    // This also avoids leaving Eigen allocations alive across MATLAB errors.

    if (!isDenseRealDouble(outputs[0]) || mxGetN(outputs[0]) != 1 ||
        mxGetM(outputs[0]) != dim()) {
      invalidOutput(
          "gtsam:CustomFactor:invalidResidual",
          "CustomFactor callback must return a full real double column residual "
          "matching the noise model dimension.");
    }

    if (jacobians) {
      // MATLAB must return one Jacobian block per key, in factor key order.
      // Using a cell array keeps the MATLAB side simple and matches existing
      // wrapper conventions for heterogeneous matrix outputs.
      if (!outputs[1] || !mxIsCell(outputs[1])) {
        invalidOutput(
            "gtsam:CustomFactor:invalidJacobians",
            "CustomFactor callback must return Jacobians as a cell array.");
      }

      const size_t expected = this->keys().size();
      if (mxGetNumberOfElements(outputs[1]) != expected) {
        invalidOutput(
            "gtsam:CustomFactor:invalidJacobianCount",
            "CustomFactor callback returned the wrong number of Jacobian blocks.");
      }

      for (size_t i = 0; i < expected; ++i) {
        const mxArray* cell = mxGetCell(outputs[1], i);
        if (!cell) {
          invalidOutput(
              "gtsam:CustomFactor:emptyJacobian",
              "CustomFactor callback returned an empty Jacobian block.");
        }
        if (!isDenseRealDouble(cell) || mxGetM(cell) != dim()) {
          invalidOutput("gtsam:CustomFactor:invalidJacobians",
                        "CustomFactor Jacobian blocks must be full real double "
                        "matrices with one row per residual component.");
        }
        const size_t valueDim = values.at(keys()[i]).dim();
        if (mxGetN(cell) != valueDim) {
          invalidOutput("gtsam:CustomFactor:invalidJacobians",
                        "CustomFactor Jacobian columns must match the "
                        "corresponding value's tangent dimension.");
        }
      }
    }
  }

  /// Invoke the MATLAB callback with an owning snapshot and release outputs.
  Vector invokeCallback(const Values& values, JacobianVector* jacobians) const {
    // Call back into MATLAB through a single registry entry point:
    //   gtsam.customFactorRegistry('invoke', callbackId, values)
    // The registry owns the original MATLAB function handle and the MATLAB
    // CustomFactor proxy object, so C++ never needs to manufacture a MATLAB
    // object for `this`.
    // Materialize before allocating the argument arrays: a missing key throws
    // through the ordinary C++ wrapper boundary without stranding mxArrays.
    mxArray* wrappedValues =
        wrap_shared_ptr(callbackValues(values), "gtsam.Values", false);

    mxArray* args[4] = {
        mxCreateString(registryFunctionName().c_str()),
        mxCreateString("invoke"),
        wrap<size_t>(callback_id_),
        wrappedValues,
    };

    mxArray* outputs[2] = {nullptr, nullptr};
    mxArray* exception =
        mexCallMATLABWithTrap(jacobians ? 2 : 1, outputs, 4, args, "feval");

    destroyCallbackArgs(args, 4);

    if (exception) {
      destroyCallbackArgs(outputs, 2);
      throwCallbackError(exception);
    }

    Vector error;

    try {
      validateOutputs(values, outputs, jacobians != nullptr);
      error = unwrap<Vector>(outputs[0]);
      if (jacobians) {
        jacobians->resize(keys().size());
        for (size_t i = 0; i < keys().size(); ++i) {
          (*jacobians)[i] = unwrap<Matrix>(mxGetCell(outputs[1], i));
        }
      }
    } catch (...) {
      destroyCallbackArgs(outputs, 2);
      throw;
    }
    destroyCallbackArgs(outputs, 2);
    return error;
  }

 public:
  /**
   * Construct a MATLAB-backed CustomFactor.
   *
   * @param noiseModel Shared noise model used by the factor.
   * @param keys Factor keys in callback Jacobian order.
   * @param callbackId Registry ID previously allocated in MATLAB.
   * @param factorKeysOnly Copy only declared keys into the owning snapshot.
   */
  MatlabCustomFactor(const noiseModel::Base::shared_ptr& noiseModel,
                     const KeyVector& keys, size_t callbackId,
                     bool factorKeysOnly = false)
      : CustomFactor(
            noiseModel, keys,
            [](const CustomFactor& factor, const Values& values,
               const JacobianVector* jacobians) {
              const auto& matlabFactor =
                  static_cast<const MatlabCustomFactor&>(factor);
              return matlabFactor.invokeCallback(
                  values, const_cast<JacobianVector*>(jacobians));
            }),
        callback_id_(callbackId),
        factor_keys_only_(factorKeysOnly)
#ifdef GTSAM_USE_TBB
        ,
        tbb_serial_guard_(std::make_shared<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, 1))
#endif
  {
    // Keep the mex module loaded for the rest of the MATLAB session once a
    // MATLAB CustomFactor exists. Without this, MATLAB can unload the wrapper
    // while callback-backed factors are still reachable through optimizer-owned
    // C++ objects, which caused unload-time crashes during debugging.
    mexLock();
  }

  ~MatlabCustomFactor() override = default;
};

}  // namespace gtsam
