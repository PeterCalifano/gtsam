/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file LeggedEstimator.h
 * @date February 2026
 * @author Frank Dellaert
 * @author Pietro Califano (joint contacts and covariance extensions)
 */

#pragma once

#include <gtsam/base/MatrixConstants.h>
#include <gtsam/geometry/ExtendedPose3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/LeftLinearEKF.h>
#include <gtsam/navigation/NavState.h>
#include <gtsam/navigation/PreintegrationParams.h>
#include <gtsam/nonlinear/BatchFixedLagSmoother.h>

#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace gtsam {

inline constexpr size_t kCorrelatedContactPointCount = 4;
inline constexpr size_t kCorrelatedContactDimension =
    3 * kCorrelatedContactPointCount;
inline constexpr size_t kCorrelatedContactVelocityDimension =
    kCorrelatedContactDimension + 3;
using CorrelatedFourPointContactCovariance =
    Eigen::Matrix<double, kCorrelatedContactDimension,
                  kCorrelatedContactDimension>;
using CorrelatedFourPointVelocityCovariance =
    Eigen::Matrix<double, kCorrelatedContactVelocityDimension,
                  kCorrelatedContactVelocityDimension>;

/**
 * Point motion relative to the body, paired with a contact position.
 * Vectors use body coordinates; angularVelocityBody is the raw gyro
 * measurement. The joint covariance is for the IMU-frame residual [position,
 * zero velocity], including their cross-covariance and any gyro uncertainty
 * supplied by the caller. Point-velocity packets are rejected while
 * full-contact initialization is pending.
 */
struct PointContactVelocityMeasurement {
  Vector3 bodyPointVelocity = Vector3::Zero();
  Vector3 angularVelocityBody = Vector3::Zero();
  Matrix6 positionVelocityCovariance = Matrix6::Identity();
};

/**
 * Zero-world-velocity information at a physical foot's frame origin.
 * Position and velocity vectors use body coordinates; angularVelocityBody is
 * raw. The IMU-frame residual covariance order is [point0, point1, point2,
 * point3, foot-origin zero velocity]. Its leading 12-D block must equal the
 * group's positionCovariance. The caller supplies gyro and position/velocity
 * correlations.
 */
struct CorrelatedFourPointVelocityMeasurement {
  Vector3 bodyPoint = Vector3::Zero();
  Vector3 bodyPointVelocity = Vector3::Zero();
  Vector3 angularVelocityBody = Vector3::Zero();
  CorrelatedFourPointVelocityCovariance positionVelocityCovariance =
      CorrelatedFourPointVelocityCovariance::Identity();
};

/// Body-frame contact measurement for one foot.
struct ContactMeasurement {
  ContactMeasurement() = default;
  ContactMeasurement(
      size_t footIndex, Vector3 measuredBodyPoint, bool isTouchdown = false,
      std::optional<Matrix3> measuredPositionCovariance = std::nullopt,
      std::optional<PointContactVelocityMeasurement> measuredPointVelocity =
          std::nullopt)
      : foot(footIndex),
        bodyPoint(std::move(measuredBodyPoint)),
        touchdown(isTouchdown),
        positionCovariance(std::move(measuredPositionCovariance)),
        pointVelocity(std::move(measuredPointVelocity)) {}

  size_t foot = 0;
  Vector3 bodyPoint = Vector3::Zero();
  /// True when this measurement corresponds to a new swing-to-stance touchdown.
  bool touchdown = false;
  /// IMU-frame position-residual covariance; defaults to the configured value.
  std::optional<Matrix3> positionCovariance;
  /// Optional zero-world-velocity point measurement with a joint 6-D
  /// covariance.
  std::optional<PointContactVelocityMeasurement> pointVelocity;
};

/**
 * One physical-foot update containing four jointly correlated point contacts.
 * Points have strictly increasing indices and the same touchdown flag. Their
 * individual covariance/velocity overrides must be empty. Covariance blocks
 * follow point order and use IMU-frame residual coordinates.
 */
struct CorrelatedFourPointContactMeasurement {
  std::array<ContactMeasurement, kCorrelatedContactPointCount> points;
  CorrelatedFourPointContactCovariance positionCovariance =
      CorrelatedFourPointContactCovariance::Identity();
  std::optional<CorrelatedFourPointVelocityMeasurement> footOriginVelocity;
};

/**
 * Optional foot-origin no-slip observation paired with a relative foot pose.
 * Covariance order is [pose rotation, pose translation, IMU-frame velocity].
 * Pose coordinates use the measured foot's right-local tangent. Angular rate
 * and relative foot-origin velocity are expressed in IMU axes.
 */
struct FootPoseVelocityMeasurement {
  Vector3 footVelocityImu = Vector3::Zero();
  Vector3 angularVelocityImu = Vector3::Zero();
  Matrix9 covariance = Matrix9::Identity();
};

/**
 * Direct FK contact pose for graph smoothers; no corner representation.
 * The caller converts base-relative FK into IMU-relative FK with the fixed
 * extrinsic. Covariance is right-local, ordered rotation then translation.
 * When velocity is present, its joint covariance replaces poseCovariance.
 */
struct FootPoseContactMeasurement {
  size_t foot = 0;
  Pose3 imuPFoot;
  Matrix6 poseCovariance = Matrix6::Identity();
  bool touchdown = false;
  std::optional<FootPoseVelocityMeasurement> velocity;
};

/// Common estimator parameters shared by all four variants.
struct LeggedEstimatorParams {
  std::shared_ptr<PreintegrationParams> preintegrationParams;
  Pose3 body_P_imu = Pose3(Rot3(), Point3(0.30, 0.0, 0.15));
  double footholdProcessSigma = 1e-4;
  double footholdInitSigma = 5e-1;
  Matrix3 contactCovariance =
      (Vector3(0.03 * 0.03, 0.03 * 0.03, 0.02 * 0.02)).asDiagonal();
  double heightPriorSigma = 0.15;
  /// Enable robust contact noise for graph-based variants.
  bool useRobustContactNoise = false;
  /// Scalar-Huber threshold for robust contact factors in graph-based variants.
  double robustContactHuberK = 2.0;
  /// Constant IMU bias removed from the raw gyroscope and accelerometer data.
  imuBias::ConstantBias imuBias;
  /// Accelerometer bias random-walk sigma used by the combined smoother.
  double biasAccRandomWalkSigma = 5e-3;
  /// Gyroscope bias random-walk sigma used by the combined smoother.
  double biasOmegaRandomWalkSigma = 1e-4;
  /// Run the one-time full-contact initializer when available.
  bool useFullContactInitialization = true;
  /// Replace a leaving foot by a fresh independent prior.
  bool marginalizeLeavingFoot = true;
};

/** Optimization engine used by the two fixed-lag estimator cores. */
enum class LeggedFixedLagEngine {
  Batch,
  Incremental,
};

/// Common runtime interface shared by all four legged estimator variants.
class GTSAM_EXPORT LeggedEstimator {
 public:
  /// Destroy the estimator interface.
  virtual ~LeggedEstimator() = default;

  /// Enable contact height priors at the supplied terrain height.
  void turnHeightPriorOn(double terrainHeight) {
    terrainHeight_ = terrainHeight;
  }

  /// Disable contact height priors.
  void turnHeightPriorOff() { terrainHeight_.reset(); }

  /// Predict the estimator forward with one IMU sample.
  virtual void predict(const Vector3& omegaBody,
                       const Vector3& specificForceBody, double dt) = 0;

  /**
   * Process the complete active contact set at the current estimator time.
   * An omitted contact ends its stance episode. An explicit touchdown starts
   * a new episode even if the contact was already active.
   */
  virtual void processContacts(
      const std::vector<ContactMeasurement>& activeContacts) = 0;

  /**
   * Process the active physical-foot groups in the EKF/IEKF variants.
   * Supply the complete active set, as for processContacts(). Point indices
   * select state entries; their order within each group selects covariance
   * rows.
   * @throws std::invalid_argument for malformed measurements or covariances,
   *         or if called on a fixed-lag smoother.
   * @throws std::out_of_range for contact indices outside the configured state.
   * @throws std::logic_error while full-contact initialization is pending.
   */
  virtual void processCorrelatedContacts(
      const std::vector<CorrelatedFourPointContactMeasurement>&
          activeContactGroups) = 0;

  /**
   * Return the current propagated estimate as `ExtendedPose3d`.
   *
   * The state layout is `(R, p, v, f_1, ..., f_k)`, where the first two
   * translational blocks are base position and base velocity, followed by one
   * world-frame foothold block per foot.
   */
  virtual ExtendedPose3d estimate() const = 0;

  /// Return the IMU bias used or estimated by the current estimator.
  virtual imuBias::ConstantBias estimateBias() const = 0;

 protected:
  /// Optional terrain height used internally for contact height priors.
  const std::optional<double>& terrainHeight() const { return terrainHeight_; }

  /// Build the shared `ExtendedPose3d` state layout used by all estimators.
  static ExtendedPose3d MakeEstimate(const NavState& navState,
                                     const Matrix& footholds);

  /// Recover the base `NavState` from an `ExtendedPose3d` estimate.
  static NavState EstimateNavState(const ExtendedPose3d& estimate) {
    return NavState(estimate.rotation(), estimate.x(0), estimate.x(1));
  }

  /// Recover the foothold matrix from an `ExtendedPose3d` estimate.
  static Matrix EstimateFootholds(const ExtendedPose3d& estimate);

 private:
  std::optional<double> terrainHeight_;
};

/**
 * Invariant filter: LeftLinearEKF on ExtendedPose3(2+k).
 *
 * The state is a single dynamic `ExtendedPose3` containing
 * `(R, p, v, f_1, ..., f_k)` in `R,p,v` order. The prediction step is
 * left-linear, and the contact measurement model has Jacobians that are
 * invariant in the state coordinates.
 *
 * This is very similar to the Contact-aided invariant filter from
 *
 *   Hartley, Ross, Maani Ghaffari, Ryan M. Eustice, and Jessy W. Grizzle.
 *   "Contact-aided invariant extended Kalman filtering for robot state
 *   estimation." The International Journal of Robotics Research 39, no. 4
 *   (2020): 402-430.
 */
class GTSAM_EXPORT LeggedInvariantEKF : public LeftLinearEKF<ExtendedPose3d>,
                                        public LeggedEstimator {
 public:
  using EkfBase = LeftLinearEKF<ExtendedPose3d>;
  using TangentVector = typename EkfBase::TangentVector;
  using Jacobian = typename EkfBase::Jacobian;
  using Covariance = typename EkfBase::Covariance;

  /// Construct the ExtendedPose3-based EKF.
  LeggedInvariantEKF(const NavState& navState0, const Matrix& footholds0,
                     const Matrix& P0, const LeggedEstimatorParams& params,
                     const std::vector<std::string>& footNames = {});

  /// Return the current full covariance.
  Matrix covariance() const { return EkfBase::covariance(); }

  /// Number of feet tracked by the estimator.
  size_t numFeet() const { return numFeet_; }

  /// Foot names in state order.
  const std::vector<std::string>& footNames() const { return footNames_; }

  /// Shared estimator parameters.
  const LeggedEstimatorParams& params() const { return params_; }

  /// Current estimate in the shared `ExtendedPose3d` layout.
  ExtendedPose3d estimate() const override {
    return MakeEstimate(baseState(), footholdMatrix());
  }

  /// Return the fixed IMU bias used by the filter.
  imuBias::ConstantBias estimateBias() const override {
    return params_.imuBias;
  }

  /// Predict forward using one IMU sample.
  void predict(const Vector3& omegaBody, const Vector3& specificForceBody,
               double dt) override;

  /// Process all currently active contacts for the current step.
  void processContacts(
      const std::vector<ContactMeasurement>& activeContacts) override;

  void processCorrelatedContacts(
      const std::vector<CorrelatedFourPointContactMeasurement>&
          activeContactGroups) override;

  /// Autonomous flow used by the left-linear prediction step.
  struct AutonomousFlow {
    /// Construct the autonomous-flow functor.
    explicit AutonomousFlow(size_t numFeet, double dt)
        : numFeet(numFeet), dt(dt) {}

    /// Return the differential of the flow at the identity.
    Matrix dIdentity() const {
      Matrix Phi = Matrix::Identity(9 + 3 * static_cast<int>(numFeet),
                                    9 + 3 * static_cast<int>(numFeet));
      Phi.block(3, 6, 3, 3) = I_3x3 * dt;
      return Phi;
    }

    /// Advance the position block while keeping velocity and footholds fixed.
    ExtendedPose3d operator()(const ExtendedPose3d& state) const {
      Matrix blocks = state.xMatrix();
      blocks.col(0) += blocks.col(1) * dt;
      return ExtendedPose3d(state.rotation(), blocks);
    }
    size_t numFeet;
    double dt;
  };

  /// Build an `ExtendedPose3(2+k)` state from base state and world footholds.
  static ExtendedPose3d MakeState(const NavState& navState,
                                  const Matrix& footholds);

  /// Build the gravity-only left increment for one prediction step.
  static ExtendedPose3d GravityIncrement(size_t numFeet, const Vector3& gravity,
                                         double dt);

  /// Build the IMU-only right increment for one prediction step.
  static ExtendedPose3d ImuIncrement(size_t numFeet, const Vector3& omegaBody,
                                     const Vector3& specificForceBody,
                                     double dt);

  /// Return the `ExtendedPose3` block index corresponding to a foot number.
  static size_t FootColumn(size_t foot) { return 2 + foot; }

 protected:
  NavState baseState() const {
    return {this->X_.rotation(), this->X_.x(0), this->X_.x(1)};
  }
  Matrix footholdMatrix() const {
    return this->X_.xMatrix().rightCols(static_cast<Eigen::Index>(numFeet()));
  }
  void resetFootToMeasurement(size_t foot, const Vector3& bodyPoint);
  void marginalizeFoot(size_t foot);
  virtual void applyContactUpdate(
      const std::vector<ContactMeasurement>& activeContacts);
  virtual void applyCorrelatedContactUpdate(
      const std::vector<CorrelatedFourPointContactMeasurement>&
          activeContactGroups);
  bool awaitingFullContactInitialization() const {
    return params_.useFullContactInitialization && !fullContactInitialized_;
  }

 private:
  bool maybeInitializeFromFullContact(
      const std::vector<ContactMeasurement>& activeContacts,
      const std::vector<bool>& activeFeet);
  Covariance processNoise(double dt) const;
  void processContactPacket(
      std::vector<ContactMeasurement> sortedContacts,
      const std::vector<CorrelatedFourPointContactMeasurement>*
          correlatedGroups);
  void applySingleContactUpdate(const ContactMeasurement& contact);
  void applySingleHeightPrior(size_t foot, double terrainHeight);

  size_t numFeet_;
  LeggedEstimatorParams params_;
  std::vector<std::string> footNames_;
  std::vector<bool> inContact_;
  std::vector<bool> initialized_;
  bool fullContactInitialized_ = false;
};

/**
 * Invariant filter with local graph fragment.
 *
 * This variant combines the two ideas introduced above: it uses the
 * `ExtendedPose3(2+k)` state of `LeggedInvariantEKF`, but performs the
 * measurement phase as a small nonlinear graph solve rather than as sequential
 * EKF corrections.
 */
class GTSAM_EXPORT LeggedInvariantIEKF : public LeggedInvariantEKF {
 public:
  /// Construct the graph-update ExtendedPose3 estimator.
  LeggedInvariantIEKF(const NavState& navState0, const Matrix& footholds0,
                      const Matrix& P0, const LeggedEstimatorParams& params,
                      const std::vector<std::string>& footNames = {});

 protected:
  void applyContactUpdate(
      const std::vector<ContactMeasurement>& activeContacts) override;
  void applyCorrelatedContactUpdate(
      const std::vector<CorrelatedFourPointContactMeasurement>&
          activeContactGroups) override;
};

/**
 * Fixed-lag smoother over NavState and contact-episode footholds.
 *
 * Uses a batch or incremental fixed-lag engine. Base states are created at
 * contact events and linked by preintegrated ImuFactor2 motion factors with one
 * shared bias variable. Touchdown creates a contact-episode landmark; active
 * landmarks stay in the window until contact ends and their timestamps expire.
 */
class GTSAM_EXPORT LeggedFixedLagSmoother : public LeggedEstimator {
 public:
  /// Construct with full NavState-local (attitude, position, velocity)
  /// covariance.
  LeggedFixedLagSmoother(
      const NavState& navState0, const Matrix& footholds0,
      const Matrix9& baseCovariance0, const LeggedEstimatorParams& params,
      double lagSeconds, const std::vector<std::string>& footNames = {},
      LeggedFixedLagEngine engine = LeggedFixedLagEngine::Batch);

  /// Destroy the fixed-lag smoother variant.
  ~LeggedFixedLagSmoother() override;

  /// Number of feet tracked by the smoother front-end.
  size_t numFeet() const { return numFeet_; }

  /// Current estimate in the shared `ExtendedPose3d` layout. Pose anchors
  /// contribute their world translations; foot attitudes are not exported.
  ExtendedPose3d estimate() const override;

  /// Return the current single shared IMU bias estimate.
  imuBias::ConstantBias estimateBias() const override { return biasEstimate_; }

  /**
   * Return the 15-D joint covariance at the latest optimized contact event.
   * Order is [rotation, position, velocity, accelerometer bias, gyro bias],
   * with NavState right-local coordinates for the first nine components.
   * Includes all cross-blocks. Pending IMU samples do not propagate this
   * covariance.
   * @throws std::logic_error before graph initialization.
   * @throws std::runtime_error if covariance extraction fails or is nonfinite.
   */
  Matrix navigationStateCovariance() const;

  /// Accumulate one IMU sample and advance the dead-reckoned estimate.
  void predict(const Vector3& omegaBody, const Vector3& specificForceBody,
               double dt) override;

  /// Process the active contacts for the current step.
  void processContacts(
      const std::vector<ContactMeasurement>& activeContacts) override;

  /**
   * Process the complete active set of physical-foot poses.
   * Full-contact startup waits for all feet, retains the navigation prior,
   * and uses each FK observation once. Terrain constraints are opt-in.
   * A point/pose type change requires a new touchdown or an ended episode.
   * @param activeContacts All active physical feet; omitted feet end contact.
   * @throws std::invalid_argument for invalid measurements, covariances, or
   *         a representation change within an existing contact episode.
   */
  void processPoseContacts(
      const std::vector<FootPoseContactMeasurement>& activeContacts);

  /// Unsupported for smoothers; throws std::invalid_argument. Use
  /// processPoseContacts for rigid feet or processContacts for single points.
  void processCorrelatedContacts(
      const std::vector<CorrelatedFourPointContactMeasurement>&
          activeContactGroups) override;

 private:
  bool maybeInitializeFromFullContact(
      const std::vector<ContactMeasurement>& activeContacts,
      const std::vector<bool>& activeFeet);
  void processContactPacket(
      std::vector<ContactMeasurement> sortedContacts,
      const std::vector<FootPoseContactMeasurement>* poseContacts);
  void refreshEstimateFromSmoother();
  NavState currentBaseState() const { return optimizedBaseState_; }
  Key currentBaseKey() const { return MakeBaseKey(step_); }
  static Key MakeBaseKey(size_t step) {
    return Symbol('x', static_cast<uint64_t>(step));
  }
  static Key MakeBiasKey() { return Symbol('b', 0); }
  Key MakeFootKey(size_t foot, size_t episode) const {
    // Interleave feet so long-running contact episodes cannot share a key.
    return Symbol('f', static_cast<uint64_t>(episode * numFeet_ + foot));
  }
  bool hasPendingImu() const { return pim_.deltaTij() > 0.0; }
  bool graphInitialized() const {
    return !params_.useFullContactInitialization || fullContactInitialized_;
  }
  bool awaitingFullContactInitialization() const {
    return params_.useFullContactInitialization && !fullContactInitialized_;
  }

  size_t numFeet_;
  LeggedEstimatorParams params_;
  std::vector<std::string> footNames_;
  Matrix initialFootholds_;
  Matrix9 baseCovariance0_;
  LeggedFixedLagEngine engine_;
  std::unique_ptr<FixedLagSmoother> smoother_;
  PreintegratedImuMeasurements pim_;
  size_t step_ = 0;
  double currentTime_ = 0.0;
  std::vector<bool> inContact_;
  std::vector<bool> initialized_;
  std::vector<size_t> footEpisodes_;
  std::vector<std::optional<Key>> activeFootKeys_;
  std::vector<bool> poseFoot_;
  NavState optimizedBaseState_;
  NavState deadReckonedState_;
  imuBias::ConstantBias biasEstimate_;
  bool fullContactInitialized_ = false;
};

/**
 * Fixed-lag smoother with CombinedImuFactor bias evolution.
 *
 * This variant keeps the same contact-episode smoother structure as
 * `LeggedFixedLagSmoother`, but replaces the `ImuFactor2` motion model with
 * event-to-event `CombinedImuFactor` links. Pose, velocity, and bias are
 * stored as separate graph variables at each contact event, and the bias is
 * allowed to evolve with a random walk between events.
 *
 * Supports the same batch and incremental fixed-lag engines.
 */
class GTSAM_EXPORT LeggedCombinedFixedLagSmoother : public LeggedEstimator {
 public:
  /// Construct with full NavState-local covariance, including pose-velocity
  /// cross blocks.
  LeggedCombinedFixedLagSmoother(
      const NavState& navState0, const Matrix& footholds0,
      const Matrix9& baseCovariance0, const LeggedEstimatorParams& params,
      double lagSeconds, const std::vector<std::string>& footNames = {},
      LeggedFixedLagEngine engine = LeggedFixedLagEngine::Batch);

  /// Destroy the combined-IMU fixed-lag smoother variant.
  ~LeggedCombinedFixedLagSmoother() override;

  /// Number of feet tracked by the smoother front-end.
  size_t numFeet() const { return numFeet_; }

  /// Current estimate in the shared `ExtendedPose3d` layout. Pose anchors
  /// contribute their world translations; foot attitudes are not exported.
  ExtendedPose3d estimate() const override;

  /// Return the current per-window bias estimate at the latest event.
  imuBias::ConstantBias estimateBias() const override { return biasEstimate_; }

  /**
   * Return the latest optimized event's 15-D joint covariance, including all
   * cross-blocks, in the same coordinates as
   * LeggedFixedLagSmoother::navigationStateCovariance(). Pending IMU samples
   * do not propagate this covariance.
   * @throws std::logic_error before graph initialization.
   * @throws std::runtime_error if covariance extraction fails or is nonfinite.
   */
  Matrix navigationStateCovariance() const;

  /// Accumulate one IMU sample and advance the dead-reckoned estimate.
  void predict(const Vector3& omegaBody, const Vector3& specificForceBody,
               double dt) override;

  /// Process the active contacts for the current step.
  void processContacts(
      const std::vector<ContactMeasurement>& activeContacts) override;

  /**
   * Process the complete active set of physical-foot poses.
   * Full-contact startup waits for all feet, retains the navigation prior,
   * and uses each FK observation once. Terrain constraints are opt-in.
   * A point/pose type change requires a new touchdown or an ended episode.
   * @param activeContacts All active physical feet; omitted feet end contact.
   * @throws std::invalid_argument for invalid measurements, covariances, or
   *         a representation change within an existing contact episode.
   */
  void processPoseContacts(
      const std::vector<FootPoseContactMeasurement>& activeContacts);

  /// Unsupported for smoothers; throws std::invalid_argument. Use
  /// processPoseContacts for rigid feet or processContacts for single points.
  void processCorrelatedContacts(
      const std::vector<CorrelatedFourPointContactMeasurement>&
          activeContactGroups) override;

 private:
  bool maybeInitializeFromFullContact(
      const std::vector<ContactMeasurement>& activeContacts,
      const std::vector<bool>& activeFeet);
  void processContactPacket(
      std::vector<ContactMeasurement> sortedContacts,
      const std::vector<FootPoseContactMeasurement>* poseContacts);
  void refreshEstimateFromSmoother();
  NavState currentBaseState() const { return optimizedBaseState_; }
  Key currentPoseKey() const { return MakePoseKey(step_); }
  Key currentVelocityKey() const { return MakeVelocityKey(step_); }
  Key currentBiasKey() const { return MakeBiasKey(step_); }
  static Key MakePoseKey(size_t step) {
    return Symbol('x', static_cast<uint64_t>(step));
  }
  static Key MakeVelocityKey(size_t step) {
    return Symbol('v', static_cast<uint64_t>(step));
  }
  static Key MakeBiasKey(size_t step) {
    return Symbol('b', static_cast<uint64_t>(step));
  }
  Key MakeFootKey(size_t foot, size_t episode) const {
    // Interleave feet so long-running contact episodes cannot share a key.
    return Symbol('f', static_cast<uint64_t>(episode * numFeet_ + foot));
  }
  bool hasPendingImu() const { return pim_.deltaTij() > 0.0; }
  bool graphInitialized() const {
    return !params_.useFullContactInitialization || fullContactInitialized_;
  }
  bool awaitingFullContactInitialization() const {
    return params_.useFullContactInitialization && !fullContactInitialized_;
  }

  size_t numFeet_;
  LeggedEstimatorParams params_;
  std::vector<std::string> footNames_;
  Matrix initialFootholds_;
  Matrix9 baseCovariance0_;
  LeggedFixedLagEngine engine_;
  std::unique_ptr<FixedLagSmoother> smoother_;
  PreintegratedCombinedMeasurements pim_;
  size_t step_ = 0;
  double currentTime_ = 0.0;
  std::vector<bool> inContact_;
  std::vector<bool> initialized_;
  std::vector<size_t> footEpisodes_;
  std::vector<std::optional<Key>> activeFootKeys_;
  std::vector<bool> poseFoot_;
  NavState optimizedBaseState_;
  NavState deadReckonedState_;
  imuBias::ConstantBias biasEstimate_;
  bool fullContactInitialized_ = false;
};

}  // namespace gtsam
