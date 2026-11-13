/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file LeggedEstimator.cpp
 * @date February 2026
 * @author Frank Dellaert
 * @author Pietro Califano (joint contacts and covariance extensions)
 */

#include <gtsam/base/Matrix.h>
#include <gtsam/base/MatrixConstants.h>
#include <gtsam/base/Testable.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/JointMarginal.h>
#include <gtsam/navigation/FootPoseFactors.h>
#include <gtsam/navigation/LeggedEstimator.h>
#include <gtsam/navigation/LeggedEstimatorFactors.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/NavStateImuEKF.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/PriorFactor.h>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace gtsam {

namespace {

using noiseModel::Diagonal;
using noiseModel::Gaussian;
using noiseModel::Isotropic;
using symbol_shorthand::X;

// The ordinary smoother stores pose and velocity in one NavState. These
// adapters retain that state layout and reuse the canonical contact residuals.
class NavStateFootPoseFactor : public NoiseModelFactorN<NavState, Pose3> {
  using Base = NoiseModelFactorN<NavState, Pose3>;
  FootPoseFactor factor_;

 public:
  NavStateFootPoseFactor(Key navKey, Key footKey, const Pose3& measurement,
                         const SharedNoiseModel& noise)
      : Base(noise, navKey, footKey),
        factor_(navKey, footKey, measurement, noise) {}

  NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<NavStateFootPoseFactor>(*this);
  }

  Vector evaluateError(const NavState& state, const Pose3& foot,
                       OptionalMatrixType H1 = nullptr,
                       OptionalMatrixType H2 = nullptr) const override {
    Matrix6 poseH, footH;
    const Vector6 error = factor_.evaluateErrorFixedSize(
        state.pose(), foot, H1 ? &poseH : nullptr, H2 ? &footH : nullptr);
    if (H1) {
      *H1 = Matrix::Zero(6, 9);
      H1->leftCols<6>() = poseH;
    }
    if (H2) *H2 = footH;
    return error;
  }
};

class NavStateFootPoseVelocityFactor
    : public NoiseModelFactorN<NavState, imuBias::ConstantBias, Pose3> {
  using Base = NoiseModelFactorN<NavState, imuBias::ConstantBias, Pose3>;
  FootPoseVelocityFactor factor_;

 public:
  NavStateFootPoseVelocityFactor(Key navKey, Key biasKey, Key footKey,
                                 const FootPoseContactMeasurement& measurement,
                                 const SharedNoiseModel& noise)
      : Base(noise, navKey, biasKey, footKey),
        factor_(navKey, navKey, biasKey, footKey, measurement.imuPFoot,
                measurement.velocity->footVelocityImu,
                measurement.velocity->angularVelocityImu, noise) {}

  NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<NavStateFootPoseVelocityFactor>(*this);
  }

  Vector evaluateError(const NavState& state, const imuBias::ConstantBias& bias,
                       const Pose3& foot, OptionalMatrixType H1 = nullptr,
                       OptionalMatrixType H2 = nullptr,
                       OptionalMatrixType H3 = nullptr) const override {
    Matrix poseH, velocityH;
    const Vector error = factor_.evaluateError(
        state.pose(), state.velocity(), bias, foot, H1 ? &poseH : nullptr,
        H1 ? &velocityH : nullptr, H2, H3);
    if (H1) {
      H1->resize(9, 9);
      H1->leftCols<6>() = poseH;
      // NavState retracts velocity in local axes; the inner factor takes a
      // world-frame Vector3 velocity.
      H1->rightCols<3>() = velocityH * state.rotation().matrix();
    }
    return error;
  }
};

double footPoseHeight(const Pose3& pose, OptionalJacobian<1, 6> H) {
  if (H) {
    H->leftCols<3>().setZero();
    H->rightCols<3>() = pose.rotation().matrix().row(2);
  }
  return pose.z();
}

std::vector<ContactMeasurement> validatePoseContactPacket(
    std::vector<FootPoseContactMeasurement>& contacts, size_t numFeet) {
  std::sort(contacts.begin(), contacts.end(),
            [](const auto& a, const auto& b) { return a.foot < b.foot; });

  std::vector<ContactMeasurement> episodes;
  episodes.reserve(contacts.size());
  for (const auto& contact : contacts) {
    if (contact.foot >= numFeet ||
        (!episodes.empty() && episodes.back().foot == contact.foot)) {
      throw std::invalid_argument(
          "processPoseContacts: foot indices must be unique and in range.");
    }
    if (!contact.imuPFoot.matrix().allFinite() ||
        (contact.velocity &&
         (!contact.velocity->footVelocityImu.allFinite() ||
          !contact.velocity->angularVelocityImu.allFinite()))) {
      throw std::invalid_argument(
          "processPoseContacts: pose and velocity measurements must be "
          "finite.");
    }

    // Only the covariance used by the selected residual is authoritative.
    const Matrix covariance = contact.velocity
                                  ? Matrix(contact.velocity->covariance)
                                  : Matrix(contact.poseCovariance);
    if (!covariance.allFinite() ||
        !covariance.isApprox(covariance.transpose(), 1e-12) ||
        Eigen::LLT<Matrix>(covariance).info() != Eigen::Success) {
      throw std::invalid_argument(
          "processPoseContacts: covariance must be finite, symmetric, and "
          "positive definite.");
    }
    episodes.emplace_back(contact.foot, Vector3::Zero(), contact.touchdown);
  }
  return episodes;
}

void addNavigationPrior(NonlinearFactorGraph& factors, Key poseKey,
                        Key velocityKey, const NavState& state,
                        const Matrix9& covariance) {
  // Join the separate graph keys only when evaluating the prior. A single
  // NavState residual preserves pose-velocity cross covariance in its local
  // tangent frame.
  const Expression<NavState> navigationState(NavState::FromPoseVelocity,
                                             Expression<Pose3>(poseKey),
                                             Expression<Vector3>(velocityKey));
  // Use the residual and Jacobian of PriorFactor<NavState>, so the split-state
  // smoother matches the NavState smoother in either chart-Jacobian build.
  auto priorError = [state](const NavState& estimate,
                            OptionalJacobian<9, 9> H) -> Vector9 {
#ifdef GTSAM_SLOW_BUT_CORRECT_BETWEENFACTOR
    const Vector9 error = -estimate.localCoordinates(state, H);
    if (H) *H = -*H;
    return error;
#else
    if (H) H->setIdentity();
    return -estimate.localCoordinates(state);
#endif
  };
  factors.emplace_shared<ExpressionFactor<Vector9>>(
      Gaussian::Covariance(covariance), Vector9::Zero(),
      Expression<Vector9>(priorError, navigationState));
}

Key initializationFootKey(size_t foot) {
  return Symbol('f', static_cast<uint64_t>(foot));
}

std::vector<std::string> defaultFootNames(size_t numFeet) {
  std::vector<std::string> names;
  names.reserve(numFeet);
  for (size_t foot = 0; foot < numFeet; ++foot) {
    names.push_back("foot_" + std::to_string(foot));
  }
  return names;
}

void throwIfInvalidContactCovariance(const Matrix3& covariance,
                                     const char* context) {
  if (!covariance.allFinite()) {
    throw std::invalid_argument(std::string(context) +
                                ": contact covariance must be finite.");
  }
  if (!covariance.isApprox(covariance.transpose(), 1e-12)) {
    throw std::invalid_argument(std::string(context) +
                                ": contact covariance must be symmetric.");
  }
  if (Eigen::LLT<Matrix3>(covariance).info() != Eigen::Success) {
    throw std::invalid_argument(
        std::string(context) +
        ": contact covariance must be positive definite.");
  }
}

void throwIfInvalidParams(const LeggedEstimatorParams& params) {
  if (!params.preintegrationParams) {
    throw std::invalid_argument(
        "LeggedEstimator: preintegrationParams must not be null.");
  }
  throwIfInvalidContactCovariance(params.contactCovariance, "LeggedEstimator");

  if (params.footholdInitSigma <= 0.0) {
    throw std::invalid_argument(
        "LeggedEstimator: footholdInitSigma must be positive.");
  }
  if (params.heightPriorSigma <= 0.0) {
    throw std::invalid_argument(
        "LeggedEstimator: heightPriorSigma must be positive.");
  }
  if (params.useRobustContactNoise && params.robustContactHuberK <= 0.0) {
    throw std::invalid_argument(
        "LeggedEstimator: robustContactHuberK must be positive.");
  }
  if (params.biasAccRandomWalkSigma <= 0.0 ||
      params.biasOmegaRandomWalkSigma <= 0.0) {
    throw std::invalid_argument(
        "LeggedEstimator: bias random-walk sigmas must be positive.");
  }
}

Matrix zeroFootholds(size_t numFeet) {
  Matrix footholds(3, static_cast<Eigen::Index>(numFeet));
  footholds.setZero();
  return footholds;
}

Vector3 unbiasedOmega(const LeggedEstimatorParams& params,
                      const Vector3& omegaBody) {
  return omegaBody - params.imuBias.gyroscope();
}

Vector3 unbiasedSpecificForce(const LeggedEstimatorParams& params,
                              const Vector3& specificForceBody) {
  return specificForceBody - params.imuBias.accelerometer();
}

int footBlockStart(size_t foot) { return 9 + 3 * static_cast<int>(foot); }

Matrix3 contactCovarianceFrom(
    const LeggedEstimatorParams& params,
    const std::optional<Matrix3>& overrideCovariance) {
  if (overrideCovariance) {
    return *overrideCovariance;
  }
  return params.contactCovariance;
}

void throwIfInvalidPointVelocityMeasurement(
    const PointContactVelocityMeasurement& measurement, const char* context) {
  if (!measurement.bodyPointVelocity.allFinite() ||
      !measurement.angularVelocityBody.allFinite()) {
    throw std::invalid_argument(std::string(context) +
                                ": point-velocity vectors must be finite.");
  }
  const Matrix6& covariance = measurement.positionVelocityCovariance;
  if (!covariance.allFinite()) {
    throw std::invalid_argument(
        std::string(context) +
        ": point position/velocity covariance must be finite.");
  }
  if (!covariance.isApprox(covariance.transpose(), 1e-12)) {
    throw std::invalid_argument(
        std::string(context) +
        ": point position/velocity covariance must be symmetric.");
  }
  if (Eigen::LLT<Matrix6>(covariance).info() != Eigen::Success) {
    throw std::invalid_argument(
        std::string(context) +
        ": point position/velocity covariance must be positive definite.");
  }
}

void throwIfInvalidCorrelatedContactCovariance(
    const CorrelatedFourPointContactCovariance& covariance,
    const char* context) {
  if (!covariance.allFinite()) {
    throw std::invalid_argument(std::string(context) +
                                ": correlated covariance must be finite.");
  }
  if (!covariance.isApprox(covariance.transpose(), 1e-12)) {
    throw std::invalid_argument(std::string(context) +
                                ": correlated covariance must be symmetric.");
  }
  if (Eigen::LLT<CorrelatedFourPointContactCovariance>(covariance).info() !=
      Eigen::Success) {
    throw std::invalid_argument(
        std::string(context) +
        ": correlated covariance must be positive definite.");
  }
}

void throwIfInvalidCorrelatedVelocityMeasurement(
    const CorrelatedFourPointVelocityMeasurement& measurement,
    const CorrelatedFourPointContactCovariance& positionCovariance,
    const char* context) {
  if (!measurement.bodyPoint.allFinite() ||
      !measurement.bodyPointVelocity.allFinite() ||
      !measurement.angularVelocityBody.allFinite()) {
    throw std::invalid_argument(
        std::string(context) +
        ": grouped foot-origin velocity vectors must be finite.");
  }
  const CorrelatedFourPointVelocityCovariance& covariance =
      measurement.positionVelocityCovariance;
  if (!covariance.allFinite() ||
      !covariance.isApprox(covariance.transpose(), 1e-12) ||
      Eigen::LLT<CorrelatedFourPointVelocityCovariance>(covariance).info() !=
          Eigen::Success) {
    throw std::invalid_argument(
        std::string(context) +
        ": grouped position/velocity covariance must be finite, symmetric, "
        "and positive definite.");
  }
  if (!covariance
           .topLeftCorner<kCorrelatedContactDimension,
                          kCorrelatedContactDimension>()
           .isApprox(positionCovariance, 1e-12)) {
    throw std::invalid_argument(
        std::string(context) +
        ": grouped position covariance must equal the leading block of the "
        "joint 15-D covariance.");
  }
}

SharedNoiseModel fixedLagBiasPriorModel() {
  // The contact-event smoother only constrains bias intermittently, so a tight
  // prior keeps the single persistent bias variable from absorbing large
  // translational drift between sparse contact updates.
  Vector6 sigmas{1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6};
  return Diagonal::Sigmas(sigmas);
}

LevenbergMarquardtParams leggedLmParams();

std::unique_ptr<FixedLagSmoother> makeLeggedFixedLagSmoother(
    double lagSeconds, LeggedFixedLagEngine engine) {
  if (!std::isfinite(lagSeconds) || lagSeconds <= 0.0) {
    throw std::invalid_argument(
        "Legged fixed-lag duration must be finite and positive.");
  }
  if (engine == LeggedFixedLagEngine::Batch) {
    return std::make_unique<BatchFixedLagSmoother>(lagSeconds,
                                                   leggedLmParams());
  }
  if (engine != LeggedFixedLagEngine::Incremental) {
    throw std::invalid_argument("Unsupported legged fixed-lag engine.");
  }

  // Check relinearization at every contact event and reuse factor slots freed
  // when old states leave the lag window.
  ISAM2Params params;
  params.relinearizeThreshold = 0.0;
  params.relinearizeSkip = 1;
  params.findUnusedFactorSlots = true;
  return std::make_unique<IncrementalFixedLagSmoother>(lagSeconds, params);
}

JointMarginal leggedFixedLagJointMarginal(const FixedLagSmoother& smoother,
                                          const KeyVector& keys) {
  if (const auto* incremental =
          dynamic_cast<const IncrementalFixedLagSmoother*>(&smoother)) {
    // Reuse the incremental solver's factorization for the joint marginal.
    return incremental->getISAM2().jointMarginalCovariance(keys);
  }
  const auto* batch = dynamic_cast<const BatchFixedLagSmoother*>(&smoother);
  if (batch == nullptr) {
    throw std::logic_error("Unsupported legged fixed-lag smoother engine.");
  }
  const Values values = batch->calculateEstimate();
  return Marginals(batch->getFactors(), values).jointMarginalCovariance(keys);
}

Matrix jointNavigationBiasCovariance(const FixedLagSmoother& smoother,
                                     Key navigationKey, Key biasKey) {
  const JointMarginal marginal =
      leggedFixedLagJointMarginal(smoother, {navigationKey, biasKey});

  // Extract by key so the exported order is independent of elimination order.
  // Separate marginals would discard the navigation-bias cross covariance.
  Eigen::Matrix<double, 15, 15> covariance;
  covariance.topLeftCorner(9, 9) = marginal.at(navigationKey, navigationKey);
  covariance.topRightCorner(9, 6) = marginal.at(navigationKey, biasKey);
  covariance.bottomLeftCorner(6, 9) = marginal.at(biasKey, navigationKey);
  covariance.bottomRightCorner(6, 6) = marginal.at(biasKey, biasKey);
  return 0.5 * (covariance + covariance.transpose());
}

Matrix jointPoseVelocityBiasCovariance(const FixedLagSmoother& smoother,
                                       Key poseKey, Key velocityKey,
                                       Key biasKey, const NavState& state) {
  const JointMarginal marginal =
      leggedFixedLagJointMarginal(smoother, {poseKey, velocityKey, biasKey});

  // Assemble all cross blocks in the graph coordinates: local Pose3, world
  // velocity, and accelerometer/gyroscope bias.
  Eigen::Matrix<double, 15, 15> separate;
  constexpr std::array<Eigen::Index, 3> offsets{0, 6, 9};
  constexpr std::array<Eigen::Index, 3> dimensions{6, 3, 6};
  const std::array<Key, 3> keys{poseKey, velocityKey, biasKey};
  for (size_t row = 0; row < keys.size(); ++row) {
    for (size_t column = 0; column < keys.size(); ++column) {
      separate.block(offsets[row], offsets[column], dimensions[row],
                     dimensions[column]) = marginal.at(keys[row], keys[column]);
    }
  }

  // Convert to the NavState tangent frame, including its local velocity error.
  // Transform both covariance axes to retain cross correlations with bias.
  Matrix96 navState_H_pose;
  Matrix93 navState_H_velocity;
  NavState::FromPoseVelocity(state.pose(), state.velocity(), navState_H_pose,
                             navState_H_velocity);
  Eigen::Matrix<double, 15, 15> transform =
      Eigen::Matrix<double, 15, 15>::Zero();
  transform.block<9, 6>(0, 0) = navState_H_pose;
  transform.block<9, 3>(0, 6) = navState_H_velocity;
  transform.bottomRightCorner<6, 6>() = I_6x6;
  const Eigen::Matrix<double, 15, 15> covariance =
      transform * separate * transform.transpose();
  return 0.5 * (covariance + covariance.transpose());
}

void throwIfInvalidNavigationStateCovariance(const Matrix& covariance,
                                             const char* context) {
  if (covariance.rows() != 15 || covariance.cols() != 15 ||
      !covariance.allFinite() ||
      !covariance.isApprox(covariance.transpose(), 1e-10)) {
    throw std::runtime_error(std::string(context) +
                             ": invalid navigation-state covariance.");
  }
}

std::shared_ptr<PreintegrationCombinedParams> combinedPreintegrationParams(
    const LeggedEstimatorParams& params) {
  throwIfInvalidParams(params);
  auto combinedParams = std::dynamic_pointer_cast<PreintegrationCombinedParams>(
      params.preintegrationParams);
  if (combinedParams) {
    // Bias-noise overrides belong to this smoother, not other users of the
    // input.
    combinedParams =
        std::make_shared<PreintegrationCombinedParams>(*combinedParams);
  } else {
    combinedParams = std::make_shared<PreintegrationCombinedParams>(
        params.preintegrationParams->n_gravity);
    combinedParams->gyroscopeCovariance =
        params.preintegrationParams->gyroscopeCovariance;
    combinedParams->accelerometerCovariance =
        params.preintegrationParams->accelerometerCovariance;
    combinedParams->integrationCovariance =
        params.preintegrationParams->integrationCovariance;
    combinedParams->omegaCoriolis = params.preintegrationParams->omegaCoriolis;
    combinedParams->body_P_sensor = params.preintegrationParams->body_P_sensor;
  }
  combinedParams->biasAccCovariance =
      I_3x3 * (params.biasAccRandomWalkSigma * params.biasAccRandomWalkSigma);
  combinedParams->biasOmegaCovariance =
      I_3x3 *
      (params.biasOmegaRandomWalkSigma * params.biasOmegaRandomWalkSigma);
  return combinedParams;
}

LevenbergMarquardtParams leggedLmParams() {
  LevenbergMarquardtParams params;
  params.lambdaInitial = 1e3;
  params.maxIterations = 50;
  return params;
}

Point3 imuMeasurement(const Pose3& body_P_imu, const Vector3& bodyPoint);

}  // namespace

namespace {

SharedNoiseModel fullContactInitializationBasePrior() {
  // Only roll, pitch, and base height should move appreciably during the
  // contact-based inverse-kinematics solve. Planar translation, yaw, and
  // velocity are treated as fixed gauge choices for initialization.
  Vector9 sigmas{1.0, 1.0, 1e-6, 1e-6, 1e-6, 1.0, 1e-6, 1e-6, 1e-6};
  return Diagonal::Sigmas(sigmas);
}

void validateContactPacket(
    const std::vector<ContactMeasurement>& activeContacts, size_t numFeet,
    const char* context) {
  std::set<size_t> seen;
  for (const ContactMeasurement& contact : activeContacts) {
    if (contact.foot >= numFeet) {
      throw std::out_of_range(std::string(context) +
                              ": foot index out of range.");
    }
    if (!seen.insert(contact.foot).second) {
      throw std::invalid_argument(std::string(context) +
                                  ": duplicate foot measurement.");
    }
    if (contact.positionCovariance) {
      throwIfInvalidContactCovariance(*contact.positionCovariance, context);
    }
    if (!contact.bodyPoint.allFinite()) {
      throw std::invalid_argument(std::string(context) +
                                  ": contact point must be finite.");
    }
    if (contact.pointVelocity) {
      if (contact.positionCovariance) {
        throw std::invalid_argument(
            std::string(context) +
            ": point-velocity contact must use its joint 6-D covariance, not "
            "a separate position covariance.");
      }
      throwIfInvalidPointVelocityMeasurement(*contact.pointVelocity, context);
    }
  }
}

void validateInitializationContacts(
    const std::vector<ContactMeasurement>& contacts) {
  if (std::any_of(contacts.begin(), contacts.end(),
                  [](const ContactMeasurement& contact) {
                    return contact.pointVelocity.has_value();
                  })) {
    throw std::logic_error(
        "Point-velocity contacts are unsupported during full-contact "
        "initialization.");
  }
}

std::vector<ContactMeasurement> validateCorrelatedContactPacket(
    const std::vector<CorrelatedFourPointContactMeasurement>& groups,
    size_t numFeet, const char* context) {
  std::vector<ContactMeasurement> contacts;
  contacts.reserve(groups.size() * kCorrelatedContactPointCount);
  std::set<size_t> seen;
  for (const CorrelatedFourPointContactMeasurement& group : groups) {
    throwIfInvalidCorrelatedContactCovariance(group.positionCovariance,
                                              context);
    if (group.footOriginVelocity) {
      throwIfInvalidCorrelatedVelocityMeasurement(
          *group.footOriginVelocity, group.positionCovariance, context);
    }
    const bool touchdown = group.points.front().touchdown;
    size_t previousFoot = 0;
    bool firstPoint = true;
    for (const ContactMeasurement& point : group.points) {
      if (point.foot >= numFeet) {
        throw std::out_of_range(std::string(context) +
                                ": contact index is out of range.");
      }
      if (!firstPoint && point.foot <= previousFoot) {
        throw std::invalid_argument(
            std::string(context) +
            ": group contact indices must be strictly increasing.");
      }
      if (!seen.insert(point.foot).second) {
        throw std::invalid_argument(std::string(context) +
                                    ": duplicate contact index.");
      }
      if (!point.bodyPoint.allFinite()) {
        throw std::invalid_argument(std::string(context) +
                                    ": contact point must be finite.");
      }
      if (point.touchdown != touchdown) {
        throw std::invalid_argument(
            std::string(context) +
            ": all points in one physical-foot group must share touchdown.");
      }
      if (point.positionCovariance || point.pointVelocity) {
        throw std::invalid_argument(
            std::string(context) +
            ": grouped points must use only the joint 12-D covariance.");
      }
      contacts.push_back(point);
      previousFoot = point.foot;
      firstPoint = false;
    }
  }
  return contacts;
}

void replaceMarginalizedFootCovariance(Matrix& covariance, size_t foot,
                                       double sigma) {
  const int dim = static_cast<int>(covariance.rows());
  const int start = footBlockStart(foot);
  std::vector<int> retainedIndices;
  retainedIndices.reserve(dim - 3);
  for (int index = 0; index < dim; ++index) {
    if (index < start || index >= start + 3) {
      retainedIndices.push_back(index);
    }
  }

  Matrix retainedCovariance(dim - 3, dim - 3);
  for (int row = 0; row < dim - 3; ++row) {
    for (int col = 0; col < dim - 3; ++col) {
      retainedCovariance(row, col) =
          covariance(retainedIndices[row], retainedIndices[col]);
    }
  }

  Matrix replacedCovariance = Matrix::Zero(dim, dim);
  for (int row = 0; row < dim - 3; ++row) {
    for (int col = 0; col < dim - 3; ++col) {
      replacedCovariance(retainedIndices[row], retainedIndices[col]) =
          retainedCovariance(row, col);
    }
  }
  replacedCovariance.block(start, start, 3, 3) = I_3x3 * (sigma * sigma);
  covariance = replacedCovariance;
}

Point3 imuMeasurement(const Pose3& body_P_imu, const Vector3& bodyPoint) {
  return body_P_imu.transformTo(Point3(bodyPoint));
}

std::array<Point3, kCorrelatedContactPointCount> imuMeasurements(
    const Pose3& body_P_imu,
    const CorrelatedFourPointContactMeasurement& group) {
  std::array<Point3, kCorrelatedContactPointCount> result;
  for (size_t point = 0; point < result.size(); ++point) {
    result.at(point) =
        imuMeasurement(body_P_imu, group.points.at(point).bodyPoint);
  }
  return result;
}

Vector3 imuVectorMeasurement(const Pose3& body_P_imu,
                             const Vector3& bodyVector) {
  return body_P_imu.rotation().unrotate(bodyVector);
}

Point3 footholdFromMeasurement(const Pose3& body_P_imu, const NavState& state,
                               const Vector3& bodyPoint) {
  return state.pose().transformFrom(imuMeasurement(body_P_imu, bodyPoint));
}

SharedNoiseModel fullCovarianceModel(const Matrix& covariance) {
  return Gaussian::Covariance(covariance);
}

SharedNoiseModel robustContactNoiseModel(const LeggedEstimatorParams& params,
                                         const Matrix& covariance) {
  if (!params.useRobustContactNoise) {
    return fullCovarianceModel(covariance);
  }
  return noiseModel::Robust::Create(
      noiseModel::mEstimator::Huber::Create(
          params.robustContactHuberK, noiseModel::mEstimator::Base::Scalar),
      fullCovarianceModel(covariance));
}

struct FullContactInitializationResult {
  NavState navState;
  Matrix footholds;
  Matrix covarianceWorldFoot;
};

struct InitializedExtendedPosePosterior {
  ExtendedPose3d state;
  Matrix covariance;
};

Matrix3 fullContactFootCovariance(const Matrix& covarianceWorldFoot,
                                  size_t foot) {
  return covarianceWorldFoot.block<3, 3>(footBlockStart(foot),
                                         footBlockStart(foot));
}

Matrix regularizedCovariance(const Matrix& covariance, double minVariance) {
  const Matrix symmetric = 0.5 * (covariance + covariance.transpose());
  Eigen::SelfAdjointEigenSolver<Matrix> solver(symmetric);
  if (solver.info() != Eigen::Success) {
    return symmetric +
           Matrix::Identity(covariance.rows(), covariance.cols()) * minVariance;
  }
  Vector eigenvalues = solver.eigenvalues();
  eigenvalues = eigenvalues.array().max(minVariance);
  return solver.eigenvectors() * eigenvalues.asDiagonal() *
         solver.eigenvectors().transpose();
}

Matrix initialSmootherCovariance(const Matrix9& baseCovariance, size_t numFeet,
                                 double footholdSigma) {
  const int dim = 9 + 3 * static_cast<int>(numFeet);
  Matrix covariance = Matrix::Zero(dim, dim);
  covariance.topLeftCorner<9, 9>() = baseCovariance;
  for (size_t foot = 0; foot < numFeet; ++foot) {
    covariance.block(footBlockStart(foot), footBlockStart(foot), 3, 3) =
        I_3x3 * (footholdSigma * footholdSigma);
  }
  return covariance;
}

NavState fullContactInitializationNavState(
    const NavState& seedNavState,
    const std::vector<ContactMeasurement>& activeContacts,
    const LeggedEstimatorParams& params, double terrainHeight) {
  Matrix imuContacts(3, static_cast<Eigen::Index>(activeContacts.size()));
  for (size_t index = 0; index < activeContacts.size(); ++index) {
    imuContacts.col(static_cast<Eigen::Index>(index)) =
        imuMeasurement(params.body_P_imu, activeContacts[index].bodyPoint);
  }

  const Vector3 centroid = imuContacts.rowwise().mean();
  const Matrix centered = imuContacts.colwise() - centroid;
  Vector3 normal =
      Eigen::JacobiSVD<Matrix>(centered, Eigen::ComputeFullU).matrixU().col(2);
  if (normal.z() < 0.0) {
    normal = -normal;
  }

  const double roll = std::atan2(normal.y(), normal.z());
  const double pitch = std::asin(-normal.x());
  const Vector3 seedRpy = seedNavState.attitude().rpy();
  const Rot3 attitude = Rot3::Ypr(seedRpy.z(), pitch, roll);

  double height = 0.0;
  for (const ContactMeasurement& contact : activeContacts) {
    const Point3 measurement =
        imuMeasurement(params.body_P_imu, contact.bodyPoint);
    height += terrainHeight - attitude.matrix().row(2).dot(measurement);
  }
  height /= static_cast<double>(activeContacts.size());

  const Point3 position(seedNavState.position().x(),
                        seedNavState.position().y(), height);
  return NavState(attitude, position, Vector3::Zero());
}

KeyVector fullContactInitializationKeys(size_t numFeet) {
  KeyVector keys;
  keys.reserve(1 + numFeet);
  keys.push_back(X(0));
  for (size_t foot = 0; foot < numFeet; ++foot) {
    keys.push_back(initializationFootKey(foot));
  }
  return keys;
}

FullContactInitializationResult solveFullContactInitialization(
    const NavState& seedNavState,
    const std::vector<ContactMeasurement>& activeContacts,
    const LeggedEstimatorParams& params, size_t numFeet, double terrainHeight) {
  NonlinearFactorGraph graph;
  Values values;
  const Key baseKey = X(0);
  graph.emplace_shared<PriorFactor<NavState>>(
      baseKey, seedNavState, fullContactInitializationBasePrior());
  const NavState initializedNavState = fullContactInitializationNavState(
      seedNavState, activeContacts, params, terrainHeight);
  values.insert(baseKey, initializedNavState);

  const SharedNoiseModel heightNoise =
      Isotropic::Sigma(1, params.heightPriorSigma);
  for (const ContactMeasurement& contact : activeContacts) {
    const SharedNoiseModel contactNoise = fullCovarianceModel(
        contactCovarianceFrom(params, contact.positionCovariance));
    const Key footKey = initializationFootKey(contact.foot);
    const Point3 foothold = footholdFromMeasurement(
        params.body_P_imu, initializedNavState, contact.bodyPoint);
    values.insert(footKey, foothold);
    graph.emplace_shared<NavStatePointContactFactor>(
        baseKey, footKey, imuMeasurement(params.body_P_imu, contact.bodyPoint),
        contactNoise);
    graph.emplace_shared<PointHeightFactor>(footKey, terrainHeight,
                                            heightNoise);
  }

  Matrix footholds = zeroFootholds(numFeet);
  for (size_t foot = 0; foot < numFeet; ++foot) {
    const auto contactIt =
        std::find_if(activeContacts.begin(), activeContacts.end(),
                     [foot](const ContactMeasurement& measurement) {
                       return measurement.foot == foot;
                     });
    if (contactIt == activeContacts.end()) {
      footholds.col(static_cast<Eigen::Index>(foot)).setZero();
      continue;
    }
    footholds.col(static_cast<Eigen::Index>(foot)) = footholdFromMeasurement(
        params.body_P_imu, initializedNavState, contactIt->bodyPoint);
  }

  const KeyVector keys = fullContactInitializationKeys(numFeet);
  Values gaugeFixedValues = values;
  for (size_t foot = 0; foot < numFeet; ++foot) {
    gaugeFixedValues.update(
        initializationFootKey(foot),
        Point3(footholds.col(static_cast<Eigen::Index>(foot))));
  }
  Marginals marginals(graph, gaugeFixedValues);

  return {initializedNavState, footholds,
          marginals.jointMarginalCovariance(keys).fullMatrix()};
}

class IkInitializer {
 public:
  IkInitializer(const LeggedEstimatorParams& params, size_t numFeet,
                double terrainHeight)
      : params_(params), numFeet_(numFeet), terrainHeight_(terrainHeight) {}

  InitializedExtendedPosePosterior fuse(
      const ExtendedPose3d& priorState, const Matrix& priorCovariance,
      const std::vector<ContactMeasurement>& activeContacts) const {
    const NavState seedNavState(priorState.rotation(), priorState.x(0),
                                priorState.x(1));
    const FullContactInitializationResult initialization =
        solveFullContactInitialization(seedNavState, activeContacts, params_,
                                       numFeet_, terrainHeight_);
    Matrix blocks(
        3, static_cast<Eigen::Index>(2 + initialization.footholds.cols()));
    blocks.col(0) = initialization.navState.position();
    blocks.col(1) = initialization.navState.velocity();
    blocks.rightCols(initialization.footholds.cols()) =
        initialization.footholds;

    // The first solve supplies the optimizer's initial guess. Use the incoming
    // prior here so the final graph counts each contact observation only once.
    NonlinearFactorGraph graph;
    Values values;
    const Key key = X(0);
    graph.emplace_shared<PriorFactor<ExtendedPose3d>>(key, priorState,
                                                      priorCovariance);
    values.insert(key,
                  ExtendedPose3d(initialization.navState.attitude(), blocks));

    const SharedNoiseModel heightNoise =
        Isotropic::Sigma(1, params_.heightPriorSigma);
    for (const ContactMeasurement& contact : activeContacts) {
      const SharedNoiseModel contactNoise = robustContactNoiseModel(
          params_, contactCovarianceFrom(params_, contact.positionCovariance));
      graph.emplace_shared<ExtendedPoseContactFactor>(
          key, LeggedInvariantEKF::FootColumn(contact.foot),
          imuMeasurement(params_.body_P_imu, contact.bodyPoint), contactNoise);
      graph.emplace_shared<ExtendedPoseHeightFactor>(
          key, LeggedInvariantEKF::FootColumn(contact.foot), terrainHeight_,
          heightNoise);
    }

    LevenbergMarquardtOptimizer optimizer(graph, values, leggedLmParams());
    const Values result = optimizer.optimize();
    Marginals marginals(graph, result);
    return {result.at<ExtendedPose3d>(key), marginals.marginalCovariance(key)};
  }

 private:
  const LeggedEstimatorParams& params_;
  size_t numFeet_;
  double terrainHeight_;
};

}  // namespace

/* ************************************************************************* */
ExtendedPose3d LeggedEstimator::MakeEstimate(const NavState& navState,
                                             const Matrix& footholds) {
  Matrix blocks(3, static_cast<Eigen::Index>(2 + footholds.cols()));
  blocks.col(0) = navState.position();
  blocks.col(1) = navState.velocity();
  blocks.rightCols(footholds.cols()) = footholds;
  return ExtendedPose3d(navState.attitude(), blocks);
}

/* ************************************************************************* */
Matrix LeggedEstimator::EstimateFootholds(const ExtendedPose3d& estimate) {
  if (estimate.k() < 2) {
    throw std::invalid_argument(
        "LeggedEstimator::EstimateFootholds: estimate must contain position "
        "and velocity blocks.");
  }
  const Eigen::Index numFeet =
      static_cast<Eigen::Index>(estimate.k() - static_cast<size_t>(2));
  return estimate.xMatrix().rightCols(numFeet);
}

/* ************************************************************************* */
LeggedInvariantEKF::LeggedInvariantEKF(
    const NavState& navState0, const Matrix& footholds0, const Matrix& P0,
    const LeggedEstimatorParams& params,
    const std::vector<std::string>& footNames)
    : EkfBase(MakeState(navState0, footholds0), P0),
      numFeet_(static_cast<size_t>(footholds0.cols())),
      params_(params),
      footNames_(footNames.empty() ? defaultFootNames(numFeet_) : footNames),
      inContact_(numFeet_, false),
      initialized_(numFeet_, false) {
  throwIfInvalidParams(params_);
  if (footNames_.size() != numFeet_) {
    throw std::invalid_argument(
        "LeggedInvariantEKF: footNames must match the number of feet.");
  }
  if (P0.rows() != static_cast<int>(MakeState(navState0, footholds0).dim()) ||
      P0.cols() != static_cast<int>(MakeState(navState0, footholds0).dim())) {
    throw std::invalid_argument(
        "LeggedInvariantEKF: covariance dimension does not match state.");
  }
  initialized_.assign(numFeet_, true);
}

/* ************************************************************************* */
void LeggedInvariantEKF::processContacts(
    const std::vector<ContactMeasurement>& activeContacts) {
  validateContactPacket(activeContacts, numFeet_,
                        "LeggedInvariantEKF::processContacts");

  std::vector<ContactMeasurement> sortedContacts = activeContacts;
  std::sort(sortedContacts.begin(), sortedContacts.end(),
            [](const ContactMeasurement& a, const ContactMeasurement& b) {
              return a.foot < b.foot;
            });

  processContactPacket(std::move(sortedContacts), nullptr);
}

/* ************************************************************************* */
void LeggedInvariantEKF::processCorrelatedContacts(
    const std::vector<CorrelatedFourPointContactMeasurement>&
        activeContactGroups) {
  if (awaitingFullContactInitialization()) {
    throw std::logic_error(
        "LeggedInvariantEKF::processCorrelatedContacts requires full-contact "
        "initialization to be disabled.");
  }
  std::vector<ContactMeasurement> sortedContacts =
      validateCorrelatedContactPacket(
          activeContactGroups, numFeet_,
          "LeggedInvariantEKF::processCorrelatedContacts");
  std::sort(sortedContacts.begin(), sortedContacts.end(),
            [](const ContactMeasurement& a, const ContactMeasurement& b) {
              return a.foot < b.foot;
            });

  processContactPacket(std::move(sortedContacts), &activeContactGroups);
}

/* ************************************************************************* */
void LeggedInvariantEKF::processContactPacket(
    std::vector<ContactMeasurement> sortedContacts,
    const std::vector<CorrelatedFourPointContactMeasurement>*
        correlatedGroups) {
  std::vector<bool> activeFeet(numFeet_, false);
  for (const ContactMeasurement& contact : sortedContacts) {
    activeFeet[contact.foot] = true;
  }

  if (awaitingFullContactInitialization()) {
    validateInitializationContacts(sortedContacts);
    (void)maybeInitializeFromFullContact(sortedContacts, activeFeet);
    return;
  }

  for (size_t foot = 0; foot < numFeet_; ++foot) {
    if (!inContact_[foot] || activeFeet[foot]) {
      continue;
    }
    if (params_.marginalizeLeavingFoot) {
      marginalizeFoot(foot);
      initialized_[foot] = false;
    }
    inContact_[foot] = false;
  }

  for (const ContactMeasurement& contact : sortedContacts) {
    if (!contact.touchdown && inContact_[contact.foot] &&
        initialized_[contact.foot]) {
      continue;
    }
    resetFootToMeasurement(contact.foot, contact.bodyPoint);
    initialized_[contact.foot] = true;
  }
  if (correlatedGroups) {
    applyCorrelatedContactUpdate(*correlatedGroups);
  } else {
    applyContactUpdate(sortedContacts);
  }
  inContact_ = activeFeet;
}

/* ************************************************************************* */
bool LeggedInvariantEKF::maybeInitializeFromFullContact(
    const std::vector<ContactMeasurement>& activeContacts,
    const std::vector<bool>& activeFeet) {
  if (!params_.useFullContactInitialization || fullContactInitialized_ ||
      numFeet_ < 3 || activeContacts.size() != numFeet_) {
    return false;
  }

  const double terrainHeightValue = terrainHeight().value_or(0.0);
  const InitializedExtendedPosePosterior initialization =
      IkInitializer(params_, numFeet_, terrainHeightValue)
          .fuse(this->X_, this->P_, activeContacts);
  this->X_ = initialization.state;
  this->P_ = initialization.covariance;
  inContact_ = activeFeet;
  std::fill(initialized_.begin(), initialized_.end(), true);
  fullContactInitialized_ = true;
  return true;
}

/* ************************************************************************* */
ExtendedPose3d LeggedInvariantEKF::MakeState(const NavState& navState,
                                             const Matrix& footholds) {
  return MakeEstimate(navState, footholds);
}

/* ************************************************************************* */
ExtendedPose3d LeggedInvariantEKF::GravityIncrement(size_t numFeet,
                                                    const Vector3& gravity,
                                                    double dt) {
  Matrix blocks = zeroFootholds(2 + numFeet);
  blocks.col(0) = gravity * (0.5 * dt * dt);
  blocks.col(1) = gravity * dt;
  return ExtendedPose3d(Rot3(), blocks);
}

/* ************************************************************************* */
ExtendedPose3d LeggedInvariantEKF::ImuIncrement(
    size_t numFeet, const Vector3& omegaBody, const Vector3& specificForceBody,
    double dt) {
  const Vector3 phiBody = omegaBody * dt;
  const so3::DexpFunctor local(phiBody);
  Matrix blocks = zeroFootholds(2 + numFeet);
  blocks.col(0) = local.Gamma().left() * specificForceBody * dt * dt;
  blocks.col(1) = local.Jacobian().left() * specificForceBody * dt;
  return ExtendedPose3d(Rot3::Expmap(phiBody), blocks);
}

/* ************************************************************************* */
LeggedInvariantEKF::Covariance LeggedInvariantEKF::processNoise(
    double dt) const {
  const int dim = 9 + 3 * static_cast<int>(numFeet());
  Covariance Q = Covariance::Zero(dim, dim);
  const auto& pim = *params().preintegrationParams;
  Q.block(0, 0, 3, 3) = pim.gyroscopeCovariance * dt;
  Q.block(3, 3, 3, 3) = pim.integrationCovariance * dt;
  Q.block(6, 6, 3, 3) = pim.accelerometerCovariance * dt;

  const Matrix3 footNoise = I_3x3 * (params().footholdProcessSigma *
                                     params().footholdProcessSigma * dt);
  for (size_t foot = 0; foot < numFeet(); ++foot) {
    Q.block(footBlockStart(foot), footBlockStart(foot), 3, 3) = footNoise;
  }
  return Q;
}

/* ************************************************************************* */
void LeggedInvariantEKF::predict(const Vector3& omegaBody,
                                 const Vector3& specificForceBody, double dt) {
  if (dt <= 0.0) {
    throw std::invalid_argument(
        "LeggedInvariantEKF::predict: dt must be positive.");
  }
  if (awaitingFullContactInitialization()) {
    return;
  }

  const Vector3 correctedOmegaBody = unbiasedOmega(params(), omegaBody);
  const Vector3 correctedSpecificForceBody =
      unbiasedSpecificForce(params(), specificForceBody);
  const ExtendedPose3d W =
      GravityIncrement(numFeet(), params().preintegrationParams->n_gravity, dt);
  const ExtendedPose3d U = ImuIncrement(numFeet(), correctedOmegaBody,
                                        correctedSpecificForceBody, dt);
  const AutonomousFlow phi(numFeet(), dt);
  EkfBase::predict(W, phi, U, processNoise(dt));
}

/* ************************************************************************* */
void LeggedInvariantEKF::resetFootToMeasurement(size_t foot,
                                                const Vector3& bodyPoint) {
  Matrix blocks = this->X_.xMatrix();
  blocks.col(static_cast<Eigen::Index>(FootColumn(foot))) =
      footholdFromMeasurement(params().body_P_imu, baseState(), bodyPoint);
  this->X_ = ExtendedPose3d(this->X_.rotation(), blocks);
  replaceMarginalizedFootCovariance(this->P_, foot, params().footholdInitSigma);
}

/* ************************************************************************* */
void LeggedInvariantEKF::marginalizeFoot(size_t foot) {
  Matrix blocks = this->X_.xMatrix();
  blocks.col(static_cast<Eigen::Index>(FootColumn(foot))).setZero();
  this->X_ = ExtendedPose3d(this->X_.rotation(), blocks);
  replaceMarginalizedFootCovariance(this->P_, foot, params().footholdInitSigma);
}

void LeggedInvariantEKF::applySingleContactUpdate(
    const ContactMeasurement& contact) {
  const Point3 z = imuMeasurement(params().body_P_imu, contact.bodyPoint);
  Matrix measurement_H_state;
  const Point3 prediction = extendedPoseContactPrediction(
      this->X_, FootColumn(contact.foot), &measurement_H_state);
  if (!contact.pointVelocity) {
    EkfBase::update<Point3>(
        prediction, measurement_H_state, z,
        contactCovarianceFrom(params(), contact.positionCovariance));
    return;
  }

  // A stationary contact satisfies v_imu + omega x q + q_dot = 0, with all
  // vectors expressed in the IMU frame. Correct gyro bias after frame rotation.
  const PointContactVelocityMeasurement& velocity = *contact.pointVelocity;
  const Vector3 measuredPointVelocity =
      imuVectorMeasurement(params().body_P_imu, velocity.bodyPointVelocity);
  const Vector3 correctedAngularVelocity = unbiasedOmega(
      params(),
      imuVectorMeasurement(params().body_P_imu, velocity.angularVelocityBody));
  Matrix39 velocity_H_navState;
  const Vector3 bodyVelocity = baseState().bodyVelocity(&velocity_H_navState);

  // Stack position and velocity in one update to retain their supplied noise
  // cross covariance. The velocity rows depend only on the navigation state.
  Matrix joint_H_state =
      Matrix::Zero(6, static_cast<Eigen::Index>(this->X_.dim()));
  joint_H_state.topRows<3>() = measurement_H_state;
  joint_H_state.block(3, 0, 3, 9) = velocity_H_navState;

  Vector6 jointPrediction;
  jointPrediction.head<3>() = prediction;
  jointPrediction.tail<3>() = bodyVelocity;
  Vector6 jointMeasurement;
  jointMeasurement.head<3>() = z;
  jointMeasurement.tail<3>() =
      -correctedAngularVelocity.cross(z) - measuredPointVelocity;

  EkfBase::updateWithVector(jointPrediction, joint_H_state, jointMeasurement,
                            velocity.positionVelocityCovariance);
}

/* ************************************************************************* */
void LeggedInvariantEKF::applySingleHeightPrior(size_t foot,
                                                double terrainHeight) {
  Matrix prior_H_state =
      Matrix::Zero(1, static_cast<Eigen::Index>(this->X_.dim()));
  prior_H_state.block(0, footBlockStart(foot), 1, 3) =
      this->X_.rotation().matrix().row(2);
  EkfBase::update<Vector>(
      Vector1(this->X_.x(FootColumn(foot)).z()), prior_H_state,
      Vector1(terrainHeight),
      I_1x1 * (params().heightPriorSigma * params().heightPriorSigma));
}

/* ************************************************************************* */
void LeggedInvariantEKF::applyContactUpdate(
    const std::vector<ContactMeasurement>& activeContacts) {
  for (const ContactMeasurement& contact : activeContacts) {
    if (terrainHeight()) {
      applySingleHeightPrior(contact.foot, *terrainHeight());
    }
    applySingleContactUpdate(contact);
  }
}

/* ************************************************************************* */
void LeggedInvariantEKF::applyCorrelatedContactUpdate(
    const std::vector<CorrelatedFourPointContactMeasurement>&
        activeContactGroups) {
  for (const CorrelatedFourPointContactMeasurement& group :
       activeContactGroups) {
    // Linearize every row of the joint contact update at the same state.
    if (terrainHeight()) {
      for (const ContactMeasurement& contact : group.points) {
        applySingleHeightPrior(contact.foot, *terrainHeight());
      }
    }

    const bool includesVelocity = group.footOriginVelocity.has_value();
    const Eigen::Index dimension = includesVelocity
                                       ? kCorrelatedContactVelocityDimension
                                       : kCorrelatedContactDimension;
    Vector prediction(dimension);
    Vector measurement(dimension);
    Matrix measurement_H_state =
        Matrix::Zero(dimension, static_cast<Eigen::Index>(this->X_.dim()));
    const auto measuredPoints = imuMeasurements(params().body_P_imu, group);
    for (size_t point = 0; point < kCorrelatedContactPointCount; ++point) {
      const ContactMeasurement& contact = group.points.at(point);
      Matrix point_H_state;
      const Eigen::Index row = 3 * static_cast<Eigen::Index>(point);
      prediction.segment<3>(row) = extendedPoseContactPrediction(
          this->X_, FootColumn(contact.foot), &point_H_state);
      measurement.segment<3>(row) = measuredPoints.at(point);
      measurement_H_state.middleRows(row, 3) = point_H_state;
    }
    if (includesVelocity) {
      const CorrelatedFourPointVelocityMeasurement& velocity =
          *group.footOriginVelocity;
      const Point3 measuredFootOrigin =
          imuMeasurement(params().body_P_imu, velocity.bodyPoint);
      const Vector3 measuredFootOriginVelocity =
          imuVectorMeasurement(params().body_P_imu, velocity.bodyPointVelocity);
      const Vector3 correctedAngularVelocity = unbiasedOmega(
          params(), imuVectorMeasurement(params().body_P_imu,
                                         velocity.angularVelocityBody));
      Matrix39 velocity_H_navState;
      prediction.tail<3>() = baseState().bodyVelocity(&velocity_H_navState);
      measurement.tail<3>() =
          -correctedAngularVelocity.cross(measuredFootOrigin) -
          measuredFootOriginVelocity;
      measurement_H_state.block(12, 0, 3, 9) = velocity_H_navState;
      EkfBase::updateWithVector(prediction, measurement_H_state, measurement,
                                velocity.positionVelocityCovariance);
      continue;
    }
    EkfBase::updateWithVector(prediction, measurement_H_state, measurement,
                              group.positionCovariance);
  }
}

LeggedInvariantIEKF::LeggedInvariantIEKF(
    const NavState& navState0, const Matrix& footholds0, const Matrix& P0,
    const LeggedEstimatorParams& params,
    const std::vector<std::string>& footNames)
    : LeggedInvariantEKF(navState0, footholds0, P0, params, footNames) {}

/* ************************************************************************* */
void LeggedInvariantIEKF::applyContactUpdate(
    const std::vector<ContactMeasurement>& activeContacts) {
  NonlinearFactorGraph graph;
  Values values;
  const Key key = X(0);

  graph.emplace_shared<PriorFactor<ExtendedPose3d>>(key, this->X_, this->P_);
  values.insert(key, this->X_);

  for (const ContactMeasurement& contact : activeContacts) {
    const Point3 measuredPoint =
        imuMeasurement(params().body_P_imu, contact.bodyPoint);
    if (contact.pointVelocity) {
      const PointContactVelocityMeasurement& velocity = *contact.pointVelocity;
      const SharedNoiseModel contactNoise = robustContactNoiseModel(
          params(), velocity.positionVelocityCovariance);
      graph.emplace_shared<ExtendedPosePointVelocityContactFactor>(
          key, FootColumn(contact.foot), measuredPoint,
          imuVectorMeasurement(params().body_P_imu, velocity.bodyPointVelocity),
          unbiasedOmega(params(),
                        imuVectorMeasurement(params().body_P_imu,
                                             velocity.angularVelocityBody)),
          contactNoise);
    } else {
      const SharedNoiseModel contactNoise = robustContactNoiseModel(
          params(),
          contactCovarianceFrom(params(), contact.positionCovariance));
      graph.emplace_shared<ExtendedPoseContactFactor>(
          key, FootColumn(contact.foot), measuredPoint, contactNoise);
    }
    if (terrainHeight()) {
      graph.emplace_shared<ExtendedPoseHeightFactor>(
          key, FootColumn(contact.foot), *terrainHeight(),
          Isotropic::Sigma(1, params().heightPriorSigma));
    }
  }

  const LevenbergMarquardtParams optimizerParams = leggedLmParams();
  LevenbergMarquardtOptimizer optimizer(graph, values, optimizerParams);
  const Values result = optimizer.optimize();
  Marginals marginals(graph, result);
  this->X_ = result.at<ExtendedPose3d>(key);
  this->P_ = marginals.marginalCovariance(key);
}

/* ************************************************************************* */
void LeggedInvariantIEKF::applyCorrelatedContactUpdate(
    const std::vector<CorrelatedFourPointContactMeasurement>&
        activeContactGroups) {
  NonlinearFactorGraph graph;
  Values values;
  const Key key = X(0);

  graph.emplace_shared<PriorFactor<ExtendedPose3d>>(key, this->X_, this->P_);
  values.insert(key, this->X_);

  for (const CorrelatedFourPointContactMeasurement& group :
       activeContactGroups) {
    ExtendedPoseFourPointContactFactor::FootColumns footColumns;
    for (size_t point = 0; point < kCorrelatedContactPointCount; ++point) {
      const ContactMeasurement& contact = group.points.at(point);
      footColumns.at(point) = FootColumn(contact.foot);
      if (terrainHeight()) {
        graph.emplace_shared<ExtendedPoseHeightFactor>(
            key, FootColumn(contact.foot), *terrainHeight(),
            Isotropic::Sigma(1, params().heightPriorSigma));
      }
    }
    const auto measurements = imuMeasurements(params().body_P_imu, group);
    if (group.footOriginVelocity) {
      const CorrelatedFourPointVelocityMeasurement& velocity =
          *group.footOriginVelocity;
      graph.emplace_shared<ExtendedPoseFourPointVelocityContactFactor>(
          key, footColumns, measurements,
          imuMeasurement(params().body_P_imu, velocity.bodyPoint),
          imuVectorMeasurement(params().body_P_imu, velocity.bodyPointVelocity),
          unbiasedOmega(params(),
                        imuVectorMeasurement(params().body_P_imu,
                                             velocity.angularVelocityBody)),
          robustContactNoiseModel(params(),
                                  velocity.positionVelocityCovariance));
    } else {
      graph.emplace_shared<ExtendedPoseFourPointContactFactor>(
          key, footColumns, measurements,
          robustContactNoiseModel(params(), group.positionCovariance));
    }
  }

  const LevenbergMarquardtParams optimizerParams = leggedLmParams();
  LevenbergMarquardtOptimizer optimizer(graph, values, optimizerParams);
  const Values result = optimizer.optimize();
  Marginals marginals(graph, result);
  this->X_ = result.at<ExtendedPose3d>(key);
  this->P_ = marginals.marginalCovariance(key);
}

/* ************************************************************************* */
LeggedFixedLagSmoother::LeggedFixedLagSmoother(
    const NavState& navState0, const Matrix& footholds0,
    const Matrix9& baseCovariance0, const LeggedEstimatorParams& params,
    double lagSeconds, const std::vector<std::string>& footNames,
    LeggedFixedLagEngine engine)
    : numFeet_(static_cast<size_t>(footholds0.cols())),
      params_(params),
      footNames_(footNames.empty() ? defaultFootNames(numFeet_) : footNames),
      initialFootholds_(footholds0),
      baseCovariance0_(baseCovariance0),
      engine_(engine),
      smoother_(makeLeggedFixedLagSmoother(lagSeconds, engine)),
      pim_(params.preintegrationParams, params.imuBias),
      inContact_(numFeet_, false),
      initialized_(numFeet_, false),
      footEpisodes_(numFeet_, 0),
      activeFootKeys_(numFeet_),
      poseFoot_(numFeet_, false),
      optimizedBaseState_(navState0),
      deadReckonedState_(navState0),
      biasEstimate_(params.imuBias) {
  throwIfInvalidParams(params_);
  if (footNames_.size() != numFeet_) {
    throw std::invalid_argument(
        "LeggedFixedLagSmoother: footNames must match the number of feet.");
  }

  if (params_.useFullContactInitialization) {
    return;
  }

  // Start the smoother with a single prior on the initial base state.
  NonlinearFactorGraph factors;
  Values values;
  FixedLagSmoother::KeyTimestampMap timestamps;
  const Key baseKey = MakeBaseKey(0);
  const Key biasKey = MakeBiasKey();
  factors.emplace_shared<PriorFactor<NavState>>(baseKey, navState0,
                                                baseCovariance0);
  factors.emplace_shared<PriorFactor<imuBias::ConstantBias>>(
      biasKey, biasEstimate_, fixedLagBiasPriorModel());
  values.insert(baseKey, navState0);
  values.insert(biasKey, biasEstimate_);
  timestamps[baseKey] = 0.0;
  timestamps[biasKey] = 0.0;
  smoother_->update(factors, values, timestamps);
  refreshEstimateFromSmoother();
}

/* ************************************************************************* */
LeggedFixedLagSmoother::~LeggedFixedLagSmoother() = default;

/* ************************************************************************* */
Matrix LeggedFixedLagSmoother::navigationStateCovariance() const {
  if (!graphInitialized()) {
    throw std::logic_error(
        "LeggedFixedLagSmoother covariance requires graph initialization.");
  }
  const Matrix covariance = jointNavigationBiasCovariance(
      *smoother_, currentBaseKey(), MakeBiasKey());
  throwIfInvalidNavigationStateCovariance(covariance, "LeggedFixedLagSmoother");
  return covariance;
}

/* ************************************************************************* */
ExtendedPose3d LeggedFixedLagSmoother::estimate() const {
  if (!graphInitialized()) {
    return MakeEstimate(deadReckonedState_, initialFootholds_);
  }
  const Values values = smoother_->calculateEstimate();
  Matrix footholds = zeroFootholds(numFeet_);
  for (size_t foot = 0; foot < numFeet_; ++foot) {
    if (activeFootKeys_[foot] && values.exists(*activeFootKeys_[foot])) {
      footholds.col(static_cast<Eigen::Index>(foot)) =
          poseFoot_[foot]
              ? values.at<Pose3>(*activeFootKeys_[foot]).translation()
              : values.at<Point3>(*activeFootKeys_[foot]);
    }
  }
  return MakeEstimate(deadReckonedState_, footholds);
}

/* ************************************************************************* */
void LeggedFixedLagSmoother::predict(const Vector3& omegaBody,
                                     const Vector3& specificForceBody,
                                     double dt) {
  if (dt <= 0.0) {
    throw std::invalid_argument(
        "LeggedFixedLagSmoother::predict: dt must be positive.");
  }
  if (awaitingFullContactInitialization()) {
    return;
  }

  // PreintegratedImuMeasurements expects raw IMU samples and applies the
  // current bias linearization point internally.
  pim_.integrateMeasurement(specificForceBody, omegaBody, dt);
  currentTime_ += dt;
  deadReckonedState_ = pim_.predict(optimizedBaseState_, biasEstimate_);
}

/* ************************************************************************* */
void LeggedFixedLagSmoother::processContacts(
    const std::vector<ContactMeasurement>& activeContacts) {
  validateContactPacket(activeContacts, numFeet_,
                        "LeggedFixedLagSmoother::processContacts");

  // Sort contacts once so graph construction and state bookkeeping are
  // deterministic.
  std::vector<ContactMeasurement> sortedContacts = activeContacts;
  std::sort(sortedContacts.begin(), sortedContacts.end(),
            [](const ContactMeasurement& a, const ContactMeasurement& b) {
              return a.foot < b.foot;
            });

  processContactPacket(std::move(sortedContacts), nullptr);
}

/* ************************************************************************* */
void LeggedFixedLagSmoother::processPoseContacts(
    const std::vector<FootPoseContactMeasurement>& activeContacts) {
  auto sortedPoses = activeContacts;
  auto episodes = validatePoseContactPacket(sortedPoses, numFeet_);
  processContactPacket(std::move(episodes), &sortedPoses);
}

/* ************************************************************************* */
void LeggedFixedLagSmoother::processCorrelatedContacts(
    const std::vector<CorrelatedFourPointContactMeasurement>&) {
  throw std::invalid_argument(
      "LeggedFixedLagSmoother: correlated corner measurements are "
      "filter-only; use processPoseContacts for rigid-foot graph contacts.");
}

/* ************************************************************************* */
void LeggedFixedLagSmoother::processContactPacket(
    std::vector<ContactMeasurement> sortedContacts,
    const std::vector<FootPoseContactMeasurement>* poseContacts) {
  // Mark which foot indices are active in this packet.
  std::vector<bool> activeFeet(numFeet_, false);
  for (const ContactMeasurement& contact : sortedContacts) {
    activeFeet[contact.foot] = true;
    if (activeFootKeys_[contact.foot] && !contact.touchdown &&
        poseFoot_[contact.foot] != (poseContacts != nullptr)) {
      throw std::invalid_argument(
          "Changing contact state type requires a new contact episode.");
    }
  }

  const bool initializePoses =
      awaitingFullContactInitialization() && poseContacts != nullptr;
  if (awaitingFullContactInitialization()) {
    if (!poseContacts) {
      validateInitializationContacts(sortedContacts);
      (void)maybeInitializeFromFullContact(sortedContacts, activeFeet);
      return;
    }
    if (sortedContacts.size() != numFeet_) return;
  }

  for (size_t foot = 0; foot < numFeet_; ++foot) {
    // Dropped contacts end the current episode and release its active key.
    if (inContact_[foot] && !activeFeet[foot]) {
      inContact_[foot] = false;
      initialized_[foot] = false;
      activeFootKeys_[foot].reset();
    }
  }

  if (sortedContacts.empty()) {
    // Swing-only packets update bookkeeping but do not create smoother nodes.
    return;
  }

  // Assemble all contact and terrain factors that apply at the current base
  // time.
  NonlinearFactorGraph factors;
  Values values;
  FixedLagSmoother::KeyTimestampMap timestamps;
  timestamps[MakeBiasKey()] = currentTime_;

  Key baseKey = currentBaseKey();
  NavState baseState = currentBaseState();
  if (initializePoses) {
    factors.emplace_shared<PriorFactor<NavState>>(baseKey, baseState,
                                                  baseCovariance0_);
    factors.emplace_shared<PriorFactor<imuBias::ConstantBias>>(
        MakeBiasKey(), biasEstimate_, fixedLagBiasPriorModel());
    values.insert(baseKey, baseState);
    values.insert(MakeBiasKey(), biasEstimate_);
    timestamps[baseKey] = currentTime_;
  } else if (hasPendingImu()) {
    // Close the accumulated IMU interval with a new base node at this contact
    // event.
    const Key previousBaseKey = baseKey;
    ++step_;
    baseKey = currentBaseKey();
    baseState = deadReckonedState_;
    factors.emplace_shared<ImuFactor2>(previousBaseKey, baseKey, MakeBiasKey(),
                                       pim_);
    values.insert(baseKey, baseState);
    timestamps[baseKey] = currentTime_;
  } else {
    // Multiple contact packets at the same timestamp refine the current base
    // node.
    timestamps[baseKey] = currentTime_;
  }

  for (size_t index = 0; index < sortedContacts.size(); ++index) {
    const ContactMeasurement& contact = sortedContacts[index];
    const FootPoseContactMeasurement* poseContact =
        poseContacts ? &(*poseContacts)[index] : nullptr;
    if (contact.touchdown || !activeFootKeys_[contact.foot]) {
      // A new touchdown starts a fresh landmark episode with its own smoother
      // key.
      ++footEpisodes_[contact.foot];
      const Key footKey =
          MakeFootKey(contact.foot, footEpisodes_[contact.foot]);
      activeFootKeys_[contact.foot] = footKey;
      initialized_[contact.foot] = true;
      poseFoot_[contact.foot] = poseContact != nullptr;
      if (poseContact) {
        // FK supplies an initial guess and one relative measurement. An extra
        // foot prior derived from FK would count the same observation twice.
        values.insert(footKey, baseState.pose() * poseContact->imuPFoot);
      } else {
        const Point3 foothold = footholdFromMeasurement(
            params_.body_P_imu, baseState, contact.bodyPoint);
        values.insert(footKey, foothold);
        // Touchdown reinitialization in the smoother gets an explicit point
        // prior so the new landmark episode carries the same uncertainty as the
        // filter variants after replacing a foot block.
        factors.emplace_shared<PriorFactor<Point3>>(
            footKey, foothold, Isotropic::Sigma(3, params_.footholdInitSigma));
      }
    }

    const Key footKey = *activeFootKeys_[contact.foot];
    if (poseContact) {
      if (poseContact->velocity) {
        factors.emplace_shared<NavStateFootPoseVelocityFactor>(
            baseKey, MakeBiasKey(), footKey, *poseContact,
            robustContactNoiseModel(params_,
                                    poseContact->velocity->covariance));
      } else {
        factors.emplace_shared<NavStateFootPoseFactor>(
            baseKey, footKey, poseContact->imuPFoot,
            robustContactNoiseModel(params_, poseContact->poseCovariance));
      }
    } else {
      const Point3 measuredPoint =
          imuMeasurement(params_.body_P_imu, contact.bodyPoint);

      if (contact.pointVelocity) {
        const PointContactVelocityMeasurement& velocity = *contact.pointVelocity;
        factors.emplace_shared<NavStatePointVelocityContactFactor>(
            baseKey, footKey, MakeBiasKey(), measuredPoint,
            imuVectorMeasurement(params_.body_P_imu,
                                 velocity.bodyPointVelocity),
            imuVectorMeasurement(params_.body_P_imu,
                                 velocity.angularVelocityBody),
            robustContactNoiseModel(params_,
                                    velocity.positionVelocityCovariance));
      } else {
        factors.emplace_shared<NavStatePointContactFactor>(
            baseKey, footKey, measuredPoint,
            robustContactNoiseModel(
                params_,
                contactCovarianceFrom(params_, contact.positionCovariance)));
      }
    }
    if (terrainHeight()) {
      // Terrain height is modeled as an additional unary prior on the foot
      // landmark.
      const auto noise = Isotropic::Sigma(1, params_.heightPriorSigma);
      if (poseContact) {
        factors.emplace_shared<ExpressionFactor<double>>(
            noise, *terrainHeight(),
            Expression<double>(footPoseHeight, Expression<Pose3>(footKey)));
      } else {
        factors.emplace_shared<PointHeightFactor>(footKey, *terrainHeight(),
                                                  noise);
      }
    }
    // Refresh the foot timestamp so active contact episodes stay inside the lag
    // window.
    timestamps[footKey] = currentTime_;
    inContact_[contact.foot] = true;
  }

  smoother_->update(factors, values, timestamps);
  if (initializePoses) fullContactInitialized_ = true;
  refreshEstimateFromSmoother();
  // Rebase dead reckoning on the optimized event state and bias estimate.
  pim_.resetIntegrationAndSetBias(biasEstimate_);
  deadReckonedState_ = optimizedBaseState_;
}

/* ************************************************************************* */
void LeggedFixedLagSmoother::refreshEstimateFromSmoother() {
  if (!graphInitialized()) {
    return;
  }
  const Values values = smoother_->calculateEstimate();
  if (values.exists(currentBaseKey())) {
    // Pull the latest base estimate from the smoother if the current key
    // survives the lag.
    optimizedBaseState_ = values.at<NavState>(currentBaseKey());
  }
  if (values.exists(MakeBiasKey())) {
    // Keep the preintegration bias linearization point synchronized with the
    // graph.
    biasEstimate_ = values.at<imuBias::ConstantBias>(MakeBiasKey());
  }
}

/* ************************************************************************* */
bool LeggedFixedLagSmoother::maybeInitializeFromFullContact(
    const std::vector<ContactMeasurement>& activeContacts,
    const std::vector<bool>& activeFeet) {
  if (!params_.useFullContactInitialization || fullContactInitialized_ ||
      numFeet_ < 3 || activeContacts.size() != numFeet_) {
    return false;
  }

  const double terrainHeightValue = terrainHeight().value_or(0.0);
  const InitializedExtendedPosePosterior initialization =
      IkInitializer(params_, numFeet_, terrainHeightValue)
          .fuse(MakeEstimate(deadReckonedState_, initialFootholds_),
                initialSmootherCovariance(baseCovariance0_, numFeet_,
                                          params_.footholdInitSigma),
                activeContacts);

  smoother_ = makeLeggedFixedLagSmoother(smoother_->smootherLag(), engine_);
  pim_.resetIntegrationAndSetBias(params_.imuBias);
  step_ = 0;
  currentTime_ = 0.0;
  footEpisodes_.assign(numFeet_, 0);
  activeFootKeys_.assign(numFeet_, std::nullopt);
  poseFoot_.assign(numFeet_, false);
  inContact_.assign(numFeet_, false);
  initialized_.assign(numFeet_, false);

  NonlinearFactorGraph factors;
  Values values;
  FixedLagSmoother::KeyTimestampMap timestamps;
  const Key baseKey = MakeBaseKey(0);
  const Key biasKey = MakeBiasKey();
  const NavState initializedBaseState = EstimateNavState(initialization.state);
  const Matrix initializedFootholds = EstimateFootholds(initialization.state);
  values.insert(baseKey, initializedBaseState);
  values.insert(biasKey, biasEstimate_);
  factors.emplace_shared<PriorFactor<NavState>>(
      baseKey, initializedBaseState,
      regularizedCovariance(initialization.covariance.topLeftCorner<9, 9>(),
                            1e-6));
  factors.emplace_shared<PriorFactor<imuBias::ConstantBias>>(
      biasKey, biasEstimate_, fixedLagBiasPriorModel());
  timestamps[biasKey] = 0.0;
  timestamps[baseKey] = 0.0;

  for (const ContactMeasurement& contact : activeContacts) {
    ++footEpisodes_[contact.foot];
    const Key footKey = MakeFootKey(contact.foot, footEpisodes_[contact.foot]);
    activeFootKeys_[contact.foot] = footKey;
    const Point3 foothold =
        initializedFootholds.col(static_cast<Eigen::Index>(contact.foot));
    values.insert(footKey, foothold);
    factors.emplace_shared<PriorFactor<Point3>>(
        footKey, foothold,
        Gaussian::Covariance(regularizedCovariance(
            fullContactFootCovariance(initialization.covariance, contact.foot),
            1e-6)));
    inContact_[contact.foot] = activeFeet[contact.foot];
    initialized_[contact.foot] = true;
    timestamps[footKey] = 0.0;
  }

  smoother_->update(factors, values, timestamps);
  fullContactInitialized_ = true;
  refreshEstimateFromSmoother();
  pim_.resetIntegrationAndSetBias(biasEstimate_);
  deadReckonedState_ = optimizedBaseState_;
  return true;
}

/* ************************************************************************* */
LeggedCombinedFixedLagSmoother::LeggedCombinedFixedLagSmoother(
    const NavState& navState0, const Matrix& footholds0,
    const Matrix9& baseCovariance0, const LeggedEstimatorParams& params,
    double lagSeconds, const std::vector<std::string>& footNames,
    LeggedFixedLagEngine engine)
    : numFeet_(static_cast<size_t>(footholds0.cols())),
      params_(params),
      footNames_(footNames.empty() ? defaultFootNames(numFeet_) : footNames),
      initialFootholds_(footholds0),
      baseCovariance0_(baseCovariance0),
      engine_(engine),
      smoother_(makeLeggedFixedLagSmoother(lagSeconds, engine)),
      pim_(combinedPreintegrationParams(params), params.imuBias),
      inContact_(numFeet_, false),
      initialized_(numFeet_, false),
      footEpisodes_(numFeet_, 0),
      activeFootKeys_(numFeet_),
      poseFoot_(numFeet_, false),
      optimizedBaseState_(navState0),
      deadReckonedState_(navState0),
      biasEstimate_(params.imuBias) {
  throwIfInvalidParams(params_);
  if (footNames_.size() != numFeet_) {
    throw std::invalid_argument(
        "LeggedCombinedFixedLagSmoother: footNames must match the number of "
        "feet.");
  }

  if (params_.useFullContactInitialization) {
    return;
  }

  // Start the smoother with priors on the initial pose, velocity, and bias.
  NonlinearFactorGraph factors;
  Values values;
  FixedLagSmoother::KeyTimestampMap timestamps;
  const Key poseKey = MakePoseKey(0);
  const Key velocityKey = MakeVelocityKey(0);
  const Key biasKey = MakeBiasKey(0);
  addNavigationPrior(factors, poseKey, velocityKey, navState0, baseCovariance0);
  factors.emplace_shared<PriorFactor<imuBias::ConstantBias>>(
      biasKey, biasEstimate_, fixedLagBiasPriorModel());
  values.insert(poseKey, navState0.pose());
  values.insert(velocityKey, navState0.velocity());
  values.insert(biasKey, biasEstimate_);
  timestamps[poseKey] = 0.0;
  timestamps[velocityKey] = 0.0;
  timestamps[biasKey] = 0.0;
  smoother_->update(factors, values, timestamps);
  refreshEstimateFromSmoother();
}

/* ************************************************************************* */
LeggedCombinedFixedLagSmoother::~LeggedCombinedFixedLagSmoother() = default;

/* ************************************************************************* */
Matrix LeggedCombinedFixedLagSmoother::navigationStateCovariance() const {
  if (!graphInitialized()) {
    throw std::logic_error(
        "LeggedCombinedFixedLagSmoother covariance requires graph "
        "initialization.");
  }
  const Matrix covariance = jointPoseVelocityBiasCovariance(
      *smoother_, currentPoseKey(), currentVelocityKey(), currentBiasKey(),
      optimizedBaseState_);
  throwIfInvalidNavigationStateCovariance(covariance,
                                          "LeggedCombinedFixedLagSmoother");
  return covariance;
}

/* ************************************************************************* */
ExtendedPose3d LeggedCombinedFixedLagSmoother::estimate() const {
  if (!graphInitialized()) {
    return MakeEstimate(deadReckonedState_, initialFootholds_);
  }
  const Values values = smoother_->calculateEstimate();
  Matrix footholds = zeroFootholds(numFeet_);
  for (size_t foot = 0; foot < numFeet_; ++foot) {
    if (activeFootKeys_[foot] && values.exists(*activeFootKeys_[foot])) {
      footholds.col(static_cast<Eigen::Index>(foot)) =
          poseFoot_[foot]
              ? values.at<Pose3>(*activeFootKeys_[foot]).translation()
              : values.at<Point3>(*activeFootKeys_[foot]);
    }
  }
  return MakeEstimate(deadReckonedState_, footholds);
}

/* ************************************************************************* */
void LeggedCombinedFixedLagSmoother::predict(const Vector3& omegaBody,
                                             const Vector3& specificForceBody,
                                             double dt) {
  if (dt <= 0.0) {
    throw std::invalid_argument(
        "LeggedCombinedFixedLagSmoother::predict: dt must be positive.");
  }
  if (awaitingFullContactInitialization()) {
    return;
  }

  // Combined preintegration consumes raw IMU and applies the current bias hat.
  pim_.integrateMeasurement(specificForceBody, omegaBody, dt);
  currentTime_ += dt;
  deadReckonedState_ = pim_.predict(optimizedBaseState_, biasEstimate_);
}

/* ************************************************************************* */
void LeggedCombinedFixedLagSmoother::processContacts(
    const std::vector<ContactMeasurement>& activeContacts) {
  validateContactPacket(activeContacts, numFeet_,
                        "LeggedCombinedFixedLagSmoother::processContacts");

  // Sort contacts once so graph construction and state bookkeeping are
  // deterministic.
  std::vector<ContactMeasurement> sortedContacts = activeContacts;
  std::sort(sortedContacts.begin(), sortedContacts.end(),
            [](const ContactMeasurement& a, const ContactMeasurement& b) {
              return a.foot < b.foot;
            });

  processContactPacket(std::move(sortedContacts), nullptr);
}

/* ************************************************************************* */
void LeggedCombinedFixedLagSmoother::processPoseContacts(
    const std::vector<FootPoseContactMeasurement>& activeContacts) {
  auto sortedPoses = activeContacts;
  auto episodes = validatePoseContactPacket(sortedPoses, numFeet_);
  processContactPacket(std::move(episodes), &sortedPoses);
}

/* ************************************************************************* */
void LeggedCombinedFixedLagSmoother::processCorrelatedContacts(
    const std::vector<CorrelatedFourPointContactMeasurement>&) {
  throw std::invalid_argument(
      "LeggedCombinedFixedLagSmoother: correlated corner measurements are "
      "filter-only; use processPoseContacts for rigid-foot graph contacts.");
}

/* ************************************************************************* */
void LeggedCombinedFixedLagSmoother::processContactPacket(
    std::vector<ContactMeasurement> sortedContacts,
    const std::vector<FootPoseContactMeasurement>* poseContacts) {
  // Mark which foot indices are active in this packet.
  std::vector<bool> activeFeet(numFeet_, false);
  for (const ContactMeasurement& contact : sortedContacts) {
    activeFeet[contact.foot] = true;
    if (activeFootKeys_[contact.foot] && !contact.touchdown &&
        poseFoot_[contact.foot] != (poseContacts != nullptr)) {
      throw std::invalid_argument(
          "Changing contact state type requires a new contact episode.");
    }
  }

  const bool initializePoses =
      awaitingFullContactInitialization() && poseContacts != nullptr;
  if (awaitingFullContactInitialization()) {
    if (!poseContacts) {
      validateInitializationContacts(sortedContacts);
      (void)maybeInitializeFromFullContact(sortedContacts, activeFeet);
      return;
    }
    if (sortedContacts.size() != numFeet_) return;
  }

  for (size_t foot = 0; foot < numFeet_; ++foot) {
    // Dropped contacts end the current episode and release its active key.
    if (inContact_[foot] && !activeFeet[foot]) {
      inContact_[foot] = false;
      initialized_[foot] = false;
      activeFootKeys_[foot].reset();
    }
  }

  if (sortedContacts.empty()) {
    // Swing-only packets update bookkeeping but do not create smoother nodes.
    return;
  }

  // Assemble all contact and terrain factors that apply at the current base
  // time.
  NonlinearFactorGraph factors;
  Values values;
  FixedLagSmoother::KeyTimestampMap timestamps;

  Key poseKey = currentPoseKey();
  Key velocityKey = currentVelocityKey();
  Key biasKey = currentBiasKey();
  NavState baseState = currentBaseState();
  if (initializePoses) {
    addNavigationPrior(factors, poseKey, velocityKey, baseState,
                       baseCovariance0_);
    factors.emplace_shared<PriorFactor<imuBias::ConstantBias>>(
        biasKey, biasEstimate_, fixedLagBiasPriorModel());
    values.insert(poseKey, baseState.pose());
    values.insert(velocityKey, baseState.velocity());
    values.insert(biasKey, biasEstimate_);
  } else if (hasPendingImu()) {
    // Close the accumulated IMU interval with a new state and bias at this
    // contact event.
    const Key previousPoseKey = poseKey;
    const Key previousVelocityKey = velocityKey;
    const Key previousBiasKey = biasKey;
    ++step_;
    poseKey = currentPoseKey();
    velocityKey = currentVelocityKey();
    biasKey = currentBiasKey();
    baseState = deadReckonedState_;
    factors.emplace_shared<CombinedImuFactor>(
        previousPoseKey, previousVelocityKey, poseKey, velocityKey,
        previousBiasKey, biasKey, pim_);
    values.insert(poseKey, baseState.pose());
    values.insert(velocityKey, baseState.velocity());
    values.insert(biasKey, biasEstimate_);
  }
  timestamps[poseKey] = currentTime_;
  timestamps[velocityKey] = currentTime_;
  timestamps[biasKey] = currentTime_;

  for (size_t index = 0; index < sortedContacts.size(); ++index) {
    const ContactMeasurement& contact = sortedContacts[index];
    const FootPoseContactMeasurement* poseContact =
        poseContacts ? &(*poseContacts)[index] : nullptr;
    if (contact.touchdown || !activeFootKeys_[contact.foot]) {
      // A new touchdown starts a fresh landmark episode with its own smoother
      // key.
      ++footEpisodes_[contact.foot];
      const Key footKey =
          MakeFootKey(contact.foot, footEpisodes_[contact.foot]);
      activeFootKeys_[contact.foot] = footKey;
      initialized_[contact.foot] = true;
      poseFoot_[contact.foot] = poseContact != nullptr;
      if (poseContact) {
        // FK supplies an initial guess and one relative measurement. An extra
        // foot prior derived from FK would count the same observation twice.
        values.insert(footKey, baseState.pose() * poseContact->imuPFoot);
      } else {
        const Point3 foothold = footholdFromMeasurement(
            params_.body_P_imu, baseState, contact.bodyPoint);
        values.insert(footKey, foothold);
        // Touchdown reinitialization uses the same loose point prior as the
        // filter variants' foot-block replacement.
        factors.emplace_shared<PriorFactor<Point3>>(
            footKey, foothold, Isotropic::Sigma(3, params_.footholdInitSigma));
      }
    }

    const Key footKey = *activeFootKeys_[contact.foot];
    if (poseContact) {
      if (poseContact->velocity) {
        const auto& velocity = *poseContact->velocity;
        factors.emplace_shared<FootPoseVelocityFactor>(
            poseKey, velocityKey, biasKey, footKey, poseContact->imuPFoot,
            velocity.footVelocityImu, velocity.angularVelocityImu,
            robustContactNoiseModel(params_, velocity.covariance));
      } else {
        factors.emplace_shared<FootPoseFactor>(
            poseKey, footKey, poseContact->imuPFoot,
            robustContactNoiseModel(params_, poseContact->poseCovariance));
      }
    } else {
      const Point3 measuredPoint =
          imuMeasurement(params_.body_P_imu, contact.bodyPoint);
      if (contact.pointVelocity) {
        const PointContactVelocityMeasurement& velocity =
            *contact.pointVelocity;
        factors.emplace_shared<Pose3PointVelocityContactFactor>(
            poseKey, velocityKey, footKey, biasKey, measuredPoint,
            imuVectorMeasurement(params_.body_P_imu,
                                 velocity.bodyPointVelocity),
            imuVectorMeasurement(params_.body_P_imu,
                                 velocity.angularVelocityBody),
            robustContactNoiseModel(params_,
                                    velocity.positionVelocityCovariance));
      } else {
        factors.emplace_shared<Pose3PointContactFactor>(
            poseKey, footKey, measuredPoint,
            robustContactNoiseModel(
                params_,
                contactCovarianceFrom(params_, contact.positionCovariance)));
      }
    }
    if (terrainHeight()) {
      // Terrain height is modeled as an additional unary prior on the foot
      // landmark.
      const auto noise = Isotropic::Sigma(1, params_.heightPriorSigma);
      if (poseContact) {
        factors.emplace_shared<ExpressionFactor<double>>(
            noise, *terrainHeight(),
            Expression<double>(footPoseHeight, Expression<Pose3>(footKey)));
      } else {
        factors.emplace_shared<PointHeightFactor>(footKey, *terrainHeight(),
                                                  noise);
      }
    }
    // Refresh the foot timestamp so active contact episodes stay inside the lag
    // window.
    timestamps[footKey] = currentTime_;
    inContact_[contact.foot] = true;
  }

  smoother_->update(factors, values, timestamps);
  if (initializePoses) fullContactInitialized_ = true;
  refreshEstimateFromSmoother();
  // Rebase dead reckoning on the optimized event state and bias estimate.
  pim_.resetIntegrationAndSetBias(biasEstimate_);
  deadReckonedState_ = optimizedBaseState_;
}

/* ************************************************************************* */
void LeggedCombinedFixedLagSmoother::refreshEstimateFromSmoother() {
  if (!graphInitialized()) {
    return;
  }
  const Values values = smoother_->calculateEstimate();
  if (values.exists(currentPoseKey()) && values.exists(currentVelocityKey())) {
    // Pull the latest base estimate from the smoother if the current keys
    // survive the lag.
    optimizedBaseState_ = NavState(values.at<Pose3>(currentPoseKey()),
                                   values.at<Vector3>(currentVelocityKey()));
  }
  if (values.exists(currentBiasKey())) {
    // Keep the preintegration bias linearization point synchronized with the
    // graph.
    biasEstimate_ = values.at<imuBias::ConstantBias>(currentBiasKey());
  }
}

/* ************************************************************************* */
bool LeggedCombinedFixedLagSmoother::maybeInitializeFromFullContact(
    const std::vector<ContactMeasurement>& activeContacts,
    const std::vector<bool>& activeFeet) {
  if (!params_.useFullContactInitialization || fullContactInitialized_ ||
      numFeet_ < 3 || activeContacts.size() != numFeet_) {
    return false;
  }

  const double terrainHeightValue = terrainHeight().value_or(0.0);
  const InitializedExtendedPosePosterior initialization =
      IkInitializer(params_, numFeet_, terrainHeightValue)
          .fuse(MakeEstimate(deadReckonedState_, initialFootholds_),
                initialSmootherCovariance(baseCovariance0_, numFeet_,
                                          params_.footholdInitSigma),
                activeContacts);

  smoother_ = makeLeggedFixedLagSmoother(smoother_->smootherLag(), engine_);
  pim_.resetIntegrationAndSetBias(params_.imuBias);
  step_ = 0;
  currentTime_ = 0.0;
  footEpisodes_.assign(numFeet_, 0);
  activeFootKeys_.assign(numFeet_, std::nullopt);
  poseFoot_.assign(numFeet_, false);
  inContact_.assign(numFeet_, false);
  initialized_.assign(numFeet_, false);

  NonlinearFactorGraph factors;
  Values values;
  FixedLagSmoother::KeyTimestampMap timestamps;
  const Key poseKey = MakePoseKey(0);
  const Key velocityKey = MakeVelocityKey(0);
  const Key biasKey = MakeBiasKey(0);
  const NavState initializedBaseState = EstimateNavState(initialization.state);
  const Matrix initializedFootholds = EstimateFootholds(initialization.state);
  values.insert(poseKey, initializedBaseState.pose());
  values.insert(velocityKey, initializedBaseState.velocity());
  values.insert(biasKey, biasEstimate_);
  addNavigationPrior(
      factors, poseKey, velocityKey, initializedBaseState,
      regularizedCovariance(initialization.covariance.topLeftCorner<9, 9>(),
                            1e-6));
  factors.emplace_shared<PriorFactor<imuBias::ConstantBias>>(
      biasKey, biasEstimate_, fixedLagBiasPriorModel());
  timestamps[poseKey] = 0.0;
  timestamps[velocityKey] = 0.0;
  timestamps[biasKey] = 0.0;

  for (const ContactMeasurement& contact : activeContacts) {
    ++footEpisodes_[contact.foot];
    const Key footKey = MakeFootKey(contact.foot, footEpisodes_[contact.foot]);
    activeFootKeys_[contact.foot] = footKey;
    const Point3 foothold =
        initializedFootholds.col(static_cast<Eigen::Index>(contact.foot));
    values.insert(footKey, foothold);
    factors.emplace_shared<PriorFactor<Point3>>(
        footKey, foothold,
        Gaussian::Covariance(regularizedCovariance(
            fullContactFootCovariance(initialization.covariance, contact.foot),
            1e-6)));
    inContact_[contact.foot] = activeFeet[contact.foot];
    initialized_[contact.foot] = true;
    timestamps[footKey] = 0.0;
  }

  smoother_->update(factors, values, timestamps);
  fullContactInitialized_ = true;
  refreshEstimateFromSmoother();
  pim_.resetIntegrationAndSetBias(biasEstimate_);
  deadReckonedState_ = optimizedBaseState_;
  return true;
}

}  // namespace gtsam
