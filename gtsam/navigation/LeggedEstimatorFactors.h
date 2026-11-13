/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file LeggedEstimatorFactors.h
 * @brief Contact residuals for ExtendedPose3, NavState, and split pose/velocity
 * states.
 *
 * Factor measurements and residual covariances use IMU coordinates. Position
 * residuals are R^T(f - p) - q; zero-velocity residuals are
 * R^T v + (omega - b_g) x q + q_dot. Grouped velocity factors use the foot
 * origin for q and q_dot and append one velocity residual to four positions.
 * ExtendedPose3 factors take bias-corrected angular velocity; the other
 * velocity factors estimate the gyro bias through a separate bias key.
 * @date February 2026
 * @author Frank Dellaert
 * @author Pietro Califano (joint contact factors)
 */

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/MatrixConstants.h>
#include <gtsam/navigation/LeggedEstimator.h>
#include <gtsam/nonlinear/NoiseModelFactorN.h>

#include <memory>
#include <stdexcept>

namespace gtsam {

/// Return the tangent-space start index of a foot block.
inline int leggedFootBlockStart(size_t foot) {
  return 9 + 3 * static_cast<int>(foot);
}

/**
 * Predict the IMU-frame contact vector for an ExtendedPose3 state.
 *
 * `Pose3::transformTo` returns the Jacobian with respect to the embedded
 * `(R, p)` pose as `[skew(q)  -I]`, where `q = R^T (f - p)`. The foothold
 * block in `ExtendedPose3` uses body-frame tangent coordinates, and `x(i)` has
 * component Jacobian `R`, so the foothold chain rule is
 *
 *   d q / d delta_f = (d q / d f_world) * (d f_world / d delta_f)
 *                   = R^T * R
 *                   = I.
 *
 * We therefore reuse `transformTo` only for the pose block and write the foot
 * block directly as the identity.
 */
inline Vector3 extendedPoseContactPrediction(const ExtendedPose3d& state,
                                             size_t footColumn,
                                             OptionalMatrixType H = {}) {
  Matrix36 prediction_H_pose;
  const Pose3 pose(state.rotation(), state.x(0));
  const Point3 foothold = state.x(footColumn);
  const Vector3 prediction =
      pose.transformTo(foothold, H ? &prediction_H_pose : nullptr);

  if (H) {
    H->setZero(3, static_cast<Eigen::Index>(state.dim()));
    H->block(0, 0, 3, 6) = prediction_H_pose;
    const int start = leggedFootBlockStart(footColumn - 2);
    H->block(0, start, 3, 3) = I_3x3;
  }

  return prediction;
}

/// Contact factor for the `ExtendedPose3(2+k)` graph-update variant.
class ExtendedPoseContactFactor : public NoiseModelFactorN<ExtendedPose3d> {
  using Base = NoiseModelFactorN<ExtendedPose3d>;

 public:
  using Base::evaluateError;

  /// Construct from a state key, foot column, and IMU-frame measurement.
  ExtendedPoseContactFactor(Key key, size_t footColumn,
                            const Point3& measurement,
                            const SharedNoiseModel& model)
      : Base(model, key), footColumn_(footColumn), measurement_(measurement) {}

  /// Return a deep copy.
  NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<NonlinearFactor>(
        NonlinearFactor::shared_ptr(new ExtendedPoseContactFactor(*this)));
  }

  /// Evaluate the contact residual and optional Jacobian.
  Vector evaluateError(const ExtendedPose3d& state,
                       OptionalMatrixType H) const override {
    const Vector3 prediction =
        extendedPoseContactPrediction(state, footColumn_, H);

    return prediction - measurement_;
  }

 private:
  size_t footColumn_;
  Point3 measurement_;
};

/**
 * Joint four-point contact factor for an ExtendedPose3 state.
 * Stack the four 3-D residuals in measurement order so one 12-D noise model
 * whitens them together, including correlations between contact points.
 */
class ExtendedPoseFourPointContactFactor
    : public NoiseModelFactorN<ExtendedPose3d> {
  using Base = NoiseModelFactorN<ExtendedPose3d>;

 public:
  using Base::evaluateError;
  using FootColumns = std::array<size_t, kCorrelatedContactPointCount>;
  using Measurements = std::array<Point3, kCorrelatedContactPointCount>;

  ExtendedPoseFourPointContactFactor(Key key, FootColumns footColumns,
                                     Measurements measurements,
                                     const SharedNoiseModel& model)
      : Base(model, key),
        footColumns_(std::move(footColumns)),
        measurements_(std::move(measurements)) {
    if (!model || model->dim() != kCorrelatedContactDimension) {
      throw std::invalid_argument(
          "ExtendedPoseFourPointContactFactor requires a 12-D noise model.");
    }
  }

  NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<ExtendedPoseFourPointContactFactor>(*this);
  }

  Vector evaluateError(const ExtendedPose3d& state,
                       OptionalMatrixType H) const override {
    Vector error(kCorrelatedContactDimension);
    if (H) {
      H->setZero(kCorrelatedContactDimension,
                 static_cast<Eigen::Index>(state.dim()));
    }
    for (size_t point = 0; point < kCorrelatedContactPointCount; ++point) {
      Matrix point_H_state;
      const Vector3 prediction = extendedPoseContactPrediction(
          state, footColumns_.at(point), H ? &point_H_state : nullptr);
      const Eigen::Index row = 3 * static_cast<Eigen::Index>(point);
      error.segment<3>(row) = prediction - measurements_.at(point);
      if (H) {
        H->middleRows(row, 3) = point_H_state;
      }
    }
    return error;
  }

 private:
  FootColumns footColumns_;
  Measurements measurements_;
};

/**
 * Four correlated point positions plus one foot-origin velocity residual.
 * The final three rows constrain the measured foot origin, which can differ
 * from all four contact points. Angular velocity is a bias-corrected input.
 */
class ExtendedPoseFourPointVelocityContactFactor
    : public NoiseModelFactorN<ExtendedPose3d> {
  using Base = NoiseModelFactorN<ExtendedPose3d>;

 public:
  using Base::evaluateError;
  using FootColumns = std::array<size_t, kCorrelatedContactPointCount>;
  using Measurements = std::array<Point3, kCorrelatedContactPointCount>;

  ExtendedPoseFourPointVelocityContactFactor(
      Key key, FootColumns footColumns, Measurements measurements,
      const Point3& footOriginMeasurement,
      const Vector3& measuredFootOriginVelocity,
      const Vector3& correctedAngularVelocity, const SharedNoiseModel& model)
      : Base(model, key),
        footColumns_(std::move(footColumns)),
        measurements_(std::move(measurements)),
        footOriginMeasurement_(footOriginMeasurement),
        measuredFootOriginVelocity_(measuredFootOriginVelocity),
        correctedAngularVelocity_(correctedAngularVelocity) {
    if (!model || model->dim() != kCorrelatedContactVelocityDimension) {
      throw std::invalid_argument(
          "ExtendedPoseFourPointVelocityContactFactor requires a 15-D noise "
          "model.");
    }
  }

  NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<ExtendedPoseFourPointVelocityContactFactor>(*this);
  }

  Vector evaluateError(const ExtendedPose3d& state,
                       OptionalMatrixType H) const override {
    Vector error(kCorrelatedContactVelocityDimension);
    if (H) {
      H->setZero(kCorrelatedContactVelocityDimension,
                 static_cast<Eigen::Index>(state.dim()));
    }
    for (size_t point = 0; point < kCorrelatedContactPointCount; ++point) {
      Matrix point_H_state;
      const Vector3 prediction = extendedPoseContactPrediction(
          state, footColumns_.at(point), H ? &point_H_state : nullptr);
      const Eigen::Index row = 3 * static_cast<Eigen::Index>(point);
      error.segment<3>(row) = prediction - measurements_.at(point);
      if (H) {
        H->middleRows(row, 3) = point_H_state;
      }
    }

    Matrix39 velocity_H_navState;
    const NavState navState(state.rotation(), state.x(0), state.x(1));
    const Vector3 bodyVelocity =
        navState.bodyVelocity(H ? &velocity_H_navState : nullptr);
    if (H) {
      H->block(12, 0, 3, 9) = velocity_H_navState;
    }
    error.tail<3>() = bodyVelocity +
                      correctedAngularVelocity_.cross(footOriginMeasurement_) +
                      measuredFootOriginVelocity_;
    return error;
  }

 private:
  FootColumns footColumns_;
  Measurements measurements_;
  Point3 footOriginMeasurement_;
  Vector3 measuredFootOriginVelocity_;
  Vector3 correctedAngularVelocity_;
};

/** Joint point-position and zero-world-velocity factor for ExtendedPose3. */
class ExtendedPosePointVelocityContactFactor
    : public NoiseModelFactorN<ExtendedPose3d> {
  using Base = NoiseModelFactorN<ExtendedPose3d>;

 public:
  using Base::evaluateError;

  ExtendedPosePointVelocityContactFactor(
      Key key, size_t footColumn, const Point3& measurement,
      const Vector3& measuredPointVelocity,
      const Vector3& correctedAngularVelocity, const SharedNoiseModel& model)
      : Base(model, key),
        footColumn_(footColumn),
        measurement_(measurement),
        measuredPointVelocity_(measuredPointVelocity),
        correctedAngularVelocity_(correctedAngularVelocity) {
    if (!model || model->dim() != 6) {
      throw std::invalid_argument(
          "ExtendedPosePointVelocityContactFactor requires a 6-D noise model.");
    }
  }

  NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<ExtendedPosePointVelocityContactFactor>(*this);
  }

  Vector evaluateError(const ExtendedPose3d& state,
                       OptionalMatrixType H) const override {
    Matrix position_H_state;
    const Vector3 predictedPoint = extendedPoseContactPrediction(
        state, footColumn_, H ? &position_H_state : nullptr);
    Matrix39 velocity_H_navState;
    const NavState navState(state.rotation(), state.x(0), state.x(1));
    const Vector3 bodyVelocity =
        navState.bodyVelocity(H ? &velocity_H_navState : nullptr);

    if (H) {
      H->setZero(6, static_cast<Eigen::Index>(state.dim()));
      H->topRows<3>() = position_H_state;
      H->block(3, 0, 3, 9) = velocity_H_navState;
    }

    Vector6 error;
    error.head<3>() = predictedPoint - measurement_;
    error.tail<3>() = bodyVelocity +
                      correctedAngularVelocity_.cross(measurement_) +
                      measuredPointVelocity_;
    return error;
  }

 private:
  size_t footColumn_;
  Point3 measurement_;
  Vector3 measuredPointVelocity_;
  Vector3 correctedAngularVelocity_;
};

/// Height factor for the `ExtendedPose3(2+k)` graph-update variant.
class ExtendedPoseHeightFactor : public NoiseModelFactorN<ExtendedPose3d> {
  using Base = NoiseModelFactorN<ExtendedPose3d>;

 public:
  using Base::evaluateError;

  /// Construct from a state key, foot column, and terrain height.
  ExtendedPoseHeightFactor(Key key, size_t footColumn, double terrainHeight,
                           const SharedNoiseModel& model)
      : Base(model, key),
        footColumn_(footColumn),
        terrainHeight_(terrainHeight) {}

  /// Return a deep copy.
  NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<NonlinearFactor>(
        NonlinearFactor::shared_ptr(new ExtendedPoseHeightFactor(*this)));
  }

  /// Evaluate the height residual and optional Jacobian.
  Vector evaluateError(const ExtendedPose3d& state,
                       OptionalMatrixType H) const override {
    if (H) {
      H->setZero(1, static_cast<Eigen::Index>(state.dim()));
      const Matrix3 R = state.rotation().matrix();
      const int start = leggedFootBlockStart(footColumn_ - 2);
      H->block(0, start, 1, 3) = R.row(2);
    }

    return Vector1(state.x(footColumn_).z() - terrainHeight_);
  }

 private:
  size_t footColumn_;
  double terrainHeight_;
};

/// Contact factor between a NavState and a foothold point variable.
class NavStatePointContactFactor
    : public NoiseModelFactorT<Vector3, NavState, Point3> {
  using Base = NoiseModelFactorT<Vector3, NavState, Point3>;

 public:
  using Base::evaluateError;

  /// Construct from a NavState key, foothold key, and IMU-frame measurement.
  NavStatePointContactFactor(Key navKey, Key pointKey,
                             const Point3& measurement,
                             const SharedNoiseModel& model)
      : Base(model, navKey, pointKey), measurement_(measurement) {}

  /// Return a deep copy.
  NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<NonlinearFactor>(
        NonlinearFactor::shared_ptr(new NavStatePointContactFactor(*this)));
  }

  /// Evaluate the contact residual and optional Jacobians.
  Vector3 evaluateError(const NavState& navState, const Point3& foothold,
                        OptionalMatrixType H1,
                        OptionalMatrixType H2) const override {
    Matrix36 prediction_H_pose;
    Matrix3 prediction_H_foothold;
    const Vector3 prediction = navState.pose().transformTo(
        foothold, prediction_H_pose, prediction_H_foothold);

    if (H1) {
      H1->setZero(3, 9);
      H1->block(0, 0, 3, 6) = prediction_H_pose;
    }
    if (H2) {
      *H2 = prediction_H_foothold;
    }

    return prediction - measurement_;
  }

 private:
  Point3 measurement_;
};

/** Joint point-position and zero-world-velocity factor for NavState. */
class NavStatePointVelocityContactFactor
    : public NoiseModelFactorN<NavState, Point3, imuBias::ConstantBias> {
  using Base = NoiseModelFactorN<NavState, Point3, imuBias::ConstantBias>;

 public:
  using Base::evaluateError;

  NavStatePointVelocityContactFactor(Key navKey, Key pointKey, Key biasKey,
                                     const Point3& measurement,
                                     const Vector3& measuredPointVelocity,
                                     const Vector3& measuredAngularVelocity,
                                     const SharedNoiseModel& model)
      : Base(model, navKey, pointKey, biasKey),
        measurement_(measurement),
        measuredPointVelocity_(measuredPointVelocity),
        measuredAngularVelocity_(measuredAngularVelocity) {
    if (!model || model->dim() != 6) {
      throw std::invalid_argument(
          "NavStatePointVelocityContactFactor requires a 6-D noise model.");
    }
  }

  NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<NavStatePointVelocityContactFactor>(*this);
  }

  Vector evaluateError(const NavState& navState, const Point3& foothold,
                       const imuBias::ConstantBias& bias, OptionalMatrixType H1,
                       OptionalMatrixType H2,
                       OptionalMatrixType H3) const override {
    Matrix36 position_H_pose;
    Matrix3 position_H_foothold;
    const Vector3 predictedPoint =
        navState.pose().transformTo(foothold, H1 ? &position_H_pose : nullptr,
                                    H2 ? &position_H_foothold : nullptr);
    Matrix39 velocity_H_navState;
    const Vector3 bodyVelocity =
        navState.bodyVelocity(H1 ? &velocity_H_navState : nullptr);

    if (H1) {
      H1->setZero(6, 9);
      H1->block(0, 0, 3, 6) = position_H_pose;
      H1->bottomRows<3>() = velocity_H_navState;
    }
    if (H2) {
      H2->setZero(6, 3);
      H2->topRows<3>() = position_H_foothold;
    }
    if (H3) {
      H3->setZero(6, 6);
      H3->block<3, 3>(3, 3) = skewSymmetric(measurement_);
    }

    Vector6 error;
    error.head<3>() = predictedPoint - measurement_;
    error.tail<3>() =
        bodyVelocity +
        (measuredAngularVelocity_ - bias.gyroscope()).cross(measurement_) +
        measuredPointVelocity_;
    return error;
  }

 private:
  Point3 measurement_;
  Vector3 measuredPointVelocity_;
  Vector3 measuredAngularVelocity_;
};

/// Contact factor between a Pose3 and a foothold point variable.
class Pose3PointContactFactor
    : public NoiseModelFactorT<Vector3, Pose3, Point3> {
  using Base = NoiseModelFactorT<Vector3, Pose3, Point3>;

 public:
  using Base::evaluateError;

  /// Construct from a Pose3 key, foothold key, and IMU-frame measurement.
  Pose3PointContactFactor(Key poseKey, Key pointKey, const Point3& measurement,
                          const SharedNoiseModel& model)
      : Base(model, poseKey, pointKey), measurement_(measurement) {}

  /// Return a deep copy.
  NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<NonlinearFactor>(
        NonlinearFactor::shared_ptr(new Pose3PointContactFactor(*this)));
  }

  /// Evaluate the contact residual and optional Jacobians.
  Vector3 evaluateError(const Pose3& pose, const Point3& foothold,
                        OptionalMatrixType H1,
                        OptionalMatrixType H2) const override {
    return pose.transformTo(foothold, H1, H2) - measurement_;
  }

 private:
  Point3 measurement_;
};

/** Joint point-position and zero-world-velocity factor for split navigation
 * state. */
class Pose3PointVelocityContactFactor
    : public NoiseModelFactorN<Pose3, Vector3, Point3, imuBias::ConstantBias> {
  using Base = NoiseModelFactorN<Pose3, Vector3, Point3, imuBias::ConstantBias>;

 public:
  using Base::evaluateError;

  Pose3PointVelocityContactFactor(Key poseKey, Key velocityKey, Key pointKey,
                                  Key biasKey, const Point3& measurement,
                                  const Vector3& measuredPointVelocity,
                                  const Vector3& measuredAngularVelocity,
                                  const SharedNoiseModel& model)
      : Base(model, poseKey, velocityKey, pointKey, biasKey),
        measurement_(measurement),
        measuredPointVelocity_(measuredPointVelocity),
        measuredAngularVelocity_(measuredAngularVelocity) {
    if (!model || model->dim() != 6) {
      throw std::invalid_argument(
          "Pose3PointVelocityContactFactor requires a 6-D noise model.");
    }
  }

  NonlinearFactor::shared_ptr clone() const override {
    return std::make_shared<Pose3PointVelocityContactFactor>(*this);
  }

  Vector evaluateError(const Pose3& pose, const Vector3& velocity,
                       const Point3& foothold,
                       const imuBias::ConstantBias& bias, OptionalMatrixType H1,
                       OptionalMatrixType H2, OptionalMatrixType H3,
                       OptionalMatrixType H4) const override {
    Matrix36 position_H_pose;
    Matrix3 position_H_foothold;
    const Vector3 predictedPoint =
        pose.transformTo(foothold, H1 ? &position_H_pose : nullptr,
                         H3 ? &position_H_foothold : nullptr);
    Matrix3 velocity_H_rotation;
    Matrix3 velocity_H_worldVelocity;
    const Vector3 bodyVelocity =
        pose.rotation().unrotate(velocity, H1 ? &velocity_H_rotation : nullptr,
                                 H2 ? &velocity_H_worldVelocity : nullptr);

    if (H1) {
      H1->setZero(6, 6);
      H1->topRows<3>() = position_H_pose;
      H1->block<3, 3>(3, 0) = velocity_H_rotation;
    }
    if (H2) {
      H2->setZero(6, 3);
      H2->bottomRows<3>() = velocity_H_worldVelocity;
    }
    if (H3) {
      H3->setZero(6, 3);
      H3->topRows<3>() = position_H_foothold;
    }
    if (H4) {
      H4->setZero(6, 6);
      H4->block<3, 3>(3, 3) = skewSymmetric(measurement_);
    }

    Vector6 error;
    error.head<3>() = predictedPoint - measurement_;
    error.tail<3>() =
        bodyVelocity +
        (measuredAngularVelocity_ - bias.gyroscope()).cross(measurement_) +
        measuredPointVelocity_;
    return error;
  }

 private:
  Point3 measurement_;
  Vector3 measuredPointVelocity_;
  Vector3 measuredAngularVelocity_;
};

/// Height factor on a standalone foothold point variable.
class PointHeightFactor : public NoiseModelFactorN<Point3> {
  using Base = NoiseModelFactorN<Point3>;

 public:
  using Base::evaluateError;

  /// Construct from a foothold key and terrain height.
  PointHeightFactor(Key key, double terrainHeight,
                    const SharedNoiseModel& model)
      : Base(model, key), terrainHeight_(terrainHeight) {}

  /// Return a deep copy.
  NonlinearFactor::shared_ptr clone() const override {
    return std::static_pointer_cast<NonlinearFactor>(
        NonlinearFactor::shared_ptr(new PointHeightFactor(*this)));
  }

  /// Evaluate the height residual and optional Jacobian.
  Vector evaluateError(const Point3& foothold,
                       OptionalMatrixType H) const override {
    if (H) {
      H->resize(1, 3);
      *H = Matrix13{{0.0, 0.0, 1.0}};
    }
    return Vector1(foothold.z() - terrainHeight_);
  }

 private:
  double terrainHeight_;
};

}  // namespace gtsam
