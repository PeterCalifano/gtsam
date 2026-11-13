/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 *
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file testLeggedEstimatorFactors.cpp
 * @date February 2026
 * @brief Unit tests for the legged estimator factor Jacobians.
 * @author Pietro Califano (joint contact factor regressions)
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/LeggedEstimator.h>
#include <gtsam/navigation/LeggedEstimatorFactors.h>
#include <gtsam/nonlinear/factorTesting.h>

using namespace gtsam;

namespace {

constexpr int kNumFeet = 2;
constexpr int kExtendedPoseDim = 3 + 3 * (2 + kNumFeet);
constexpr int kFourPointExtendedPoseDim =
    3 + 3 * (2 + kCorrelatedContactPointCount);

NavState sampleNavState() {
  return NavState(Rot3::RzRyRx(0.1, -0.2, 0.05), Point3(0.2, -0.4, 0.3),
                  Vector3(0.6, -0.1, 0.2));
}

Matrix sampleFootholds() {
  return Matrix{{1.2, -0.3}, {0.4, 0.8}, {-0.5, 0.1}};
}

ExtendedPose3d sampleExtendedPoseState() {
  Matrix blocks(3, 2 + kNumFeet);
  const NavState X = sampleNavState();
  blocks.col(0) = X.position();
  blocks.col(1) = X.velocity();
  blocks.rightCols(kNumFeet) = sampleFootholds();
  return ExtendedPose3d(X.attitude(), blocks);
}

std::array<Point3, kCorrelatedContactPointCount> fourPointFootholds() {
  return {Point3(0.45, 0.18, -0.55), Point3(0.45, -0.18, -0.55),
          Point3(-0.15, 0.18, -0.55), Point3(-0.15, -0.18, -0.55)};
}

std::array<Point3, kCorrelatedContactPointCount> fourPointMeasurements(
    const Pose3& pose,
    const std::array<Point3, kCorrelatedContactPointCount>& footholds) {
  std::array<Point3, kCorrelatedContactPointCount> measurements;
  for (size_t point = 0; point < measurements.size(); ++point) {
    measurements.at(point) = pose.transformTo(footholds.at(point));
  }
  return measurements;
}

ExtendedPose3d fourPointExtendedPoseState() {
  const NavState state = sampleNavState();
  const auto footholds = fourPointFootholds();
  Matrix blocks(3, 2 + kCorrelatedContactPointCount);
  blocks.col(0) = state.position();
  blocks.col(1) = state.velocity();
  for (size_t point = 0; point < footholds.size(); ++point) {
    blocks.col(static_cast<Eigen::Index>(2 + point)) = footholds.at(point);
  }
  return ExtendedPose3d(state.attitude(), blocks);
}

}  // namespace

/* ************************************************************************* */
TEST(LeggedEstimatorFactors, ExtendedPoseContactFactorJacobian) {
  const ExtendedPose3d state = sampleExtendedPoseState();
  ExtendedPoseContactFactor factor(0, 3, Point3(0.15, 0.05, -0.2),
                                   noiseModel::Unit::Create(3));

  Matrix actual_H_state;
  factor.evaluateError(state, actual_H_state);
  const Matrix expected_H_state =
      numericalDerivative11<Vector, ExtendedPose3d, kExtendedPoseDim>(
          [&](const ExtendedPose3d& x) { return factor.evaluateError(x, {}); },
          state, 1e-6);

  EXPECT(assert_equal(expected_H_state, actual_H_state, 1e-6));
}

/* ************************************************************************* */
TEST(LeggedEstimatorFactors, ExtendedPoseFourPointContactFactorJacobian) {
  const ExtendedPose3d state = fourPointExtendedPoseState();
  const auto measurements =
      fourPointMeasurements(sampleNavState().pose(), fourPointFootholds());
  ExtendedPoseFourPointContactFactor factor(
      0, {2, 3, 4, 5}, measurements,
      noiseModel::Unit::Create(kCorrelatedContactDimension));

  Matrix actual_H_state;
  const Vector error = factor.evaluateError(state, actual_H_state);
  const Matrix expected_H_state =
      numericalDerivative11<Vector, ExtendedPose3d, kFourPointExtendedPoseDim>(
          [&](const ExtendedPose3d& value) {
            return factor.evaluateError(value, {});
          },
          state, 1e-6);

  EXPECT(assert_equal(Vector::Zero(kCorrelatedContactDimension), error, 1e-12));
  EXPECT(assert_equal(expected_H_state, actual_H_state, 1e-6));
}

/* ************************************************************************* */
TEST(LeggedEstimatorFactors,
     ExtendedPoseFourPointVelocityContactFactorJacobian) {
  const ExtendedPose3d state = fourPointExtendedPoseState();
  const auto measurements =
      fourPointMeasurements(sampleNavState().pose(), fourPointFootholds());
  ExtendedPoseFourPointVelocityContactFactor factor(
      0, {2, 3, 4, 5}, measurements, Point3(0.02, 0.0, -0.48),
      Vector3(0.03, -0.02, 0.01), Vector3(0.2, -0.1, 0.4),
      noiseModel::Unit::Create(kCorrelatedContactVelocityDimension));

  Values values;
  values.insert(0, state);
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, 1e-6, 1e-6);
}

/* ************************************************************************* */
TEST(LeggedEstimatorFactors, ExtendedPosePointVelocityContactFactorJacobian) {
  const ExtendedPose3d state = sampleExtendedPoseState();
  ExtendedPosePointVelocityContactFactor factor(
      0, 3, Point3(0.15, 0.05, -0.2), Vector3(0.03, -0.02, 0.01),
      Vector3(0.2, -0.1, 0.4), noiseModel::Unit::Create(6));

  Matrix actual_H_state;
  factor.evaluateError(state, actual_H_state);
  const Matrix expected_H_state =
      numericalDerivative11<Vector, ExtendedPose3d, kExtendedPoseDim>(
          [&](const ExtendedPose3d& x) { return factor.evaluateError(x, {}); },
          state, 1e-6);

  EXPECT(assert_equal(expected_H_state, actual_H_state, 1e-6));
}

/* ************************************************************************* */
TEST(LeggedEstimatorFactors, ExtendedPoseHeightFactorJacobian) {
  const ExtendedPose3d state = sampleExtendedPoseState();
  ExtendedPoseHeightFactor factor(0, 2, -0.35, noiseModel::Unit::Create(1));

  Matrix actual_H_state;
  factor.evaluateError(state, actual_H_state);
  const Matrix expected_H_state =
      numericalDerivative11<Vector, ExtendedPose3d, kExtendedPoseDim>(
          [&](const ExtendedPose3d& x) { return factor.evaluateError(x, {}); },
          state, 1e-6);

  EXPECT(assert_equal(expected_H_state, actual_H_state, 1e-5));
}

/* ************************************************************************* */
TEST(LeggedEstimatorFactors, NavStatePointContactFactorJacobians) {
  const NavState state = sampleNavState();
  const Point3 foothold(1.2, 0.8, -0.5);
  NavStatePointContactFactor factor(0, 1, Point3(0.2, -0.1, 0.4),
                                    noiseModel::Unit::Create(3));

  Matrix actual_H_state, actual_H_foothold;
  factor.evaluateError(state, foothold, actual_H_state, actual_H_foothold);
  const auto error = [&](const NavState& x, const Point3& p) {
    return factor.evaluateError(x, p, {}, {});
  };
  const Matrix expected_H_state =
      numericalDerivative21<Vector, NavState, Point3>(error, state, foothold,
                                                      1e-6);
  const Matrix expected_H_foothold =
      numericalDerivative22<Vector, NavState, Point3>(error, state, foothold,
                                                      1e-6);

  EXPECT(assert_equal(expected_H_state, actual_H_state, 1e-6));
  EXPECT(assert_equal(expected_H_foothold, actual_H_foothold, 1e-6));
}

/* ************************************************************************* */
TEST(LeggedEstimatorFactors, NavStatePointVelocityContactFactorJacobians) {
  const NavState state = sampleNavState();
  const Point3 foothold(1.2, 0.8, -0.5);
  const imuBias::ConstantBias bias(Vector3(0.01, -0.02, 0.03),
                                   Vector3(-0.04, 0.05, -0.06));
  NavStatePointVelocityContactFactor factor(
      0, 1, 2, Point3(0.2, -0.1, 0.4), Vector3(0.03, -0.02, 0.01),
      Vector3(0.2, -0.1, 0.4), noiseModel::Unit::Create(6));

  Matrix actual_H_state, actual_H_foothold, actual_H_bias;
  factor.evaluateError(state, foothold, bias, actual_H_state, actual_H_foothold,
                       actual_H_bias);
  const auto error = [&](const NavState& x, const Point3& p,
                         const imuBias::ConstantBias& b) {
    return factor.evaluateError(x, p, b, {}, {}, {});
  };

  EXPECT(assert_equal(
      numericalDerivative31<Vector, NavState, Point3, imuBias::ConstantBias>(
          error, state, foothold, bias, 1e-6),
      actual_H_state, 1e-6));
  EXPECT(assert_equal(
      numericalDerivative32<Vector, NavState, Point3, imuBias::ConstantBias>(
          error, state, foothold, bias, 1e-6),
      actual_H_foothold, 1e-6));
  EXPECT(assert_equal(
      numericalDerivative33<Vector, NavState, Point3, imuBias::ConstantBias>(
          error, state, foothold, bias, 1e-6),
      actual_H_bias, 1e-6));
}

/* ************************************************************************* */
TEST(LeggedEstimatorFactors, Pose3PointContactFactorJacobians) {
  const Pose3 pose = sampleNavState().pose();
  const Point3 foothold(1.2, 0.8, -0.5);
  Pose3PointContactFactor factor(0, 1, Point3(0.2, -0.1, 0.4),
                                 noiseModel::Unit::Create(3));

  Matrix actual_H_pose, actual_H_foothold;
  factor.evaluateError(pose, foothold, actual_H_pose, actual_H_foothold);
  const auto error = [&](const Pose3& x, const Point3& p) {
    return factor.evaluateError(x, p, {}, {});
  };
  const Matrix expected_H_pose =
      numericalDerivative21<Vector, Pose3, Point3>(error, pose, foothold, 1e-6);
  const Matrix expected_H_foothold =
      numericalDerivative22<Vector, Pose3, Point3>(error, pose, foothold, 1e-6);

  EXPECT(assert_equal(expected_H_pose, actual_H_pose, 1e-6));
  EXPECT(assert_equal(expected_H_foothold, actual_H_foothold, 1e-6));
}

/* ************************************************************************* */
TEST(LeggedEstimatorFactors, Pose3PointVelocityContactFactorJacobians) {
  const Pose3 pose = sampleNavState().pose();
  const Vector3 velocity = sampleNavState().velocity();
  const Point3 foothold(1.2, 0.8, -0.5);
  const imuBias::ConstantBias bias(Vector3(0.01, -0.02, 0.03),
                                   Vector3(-0.04, 0.05, -0.06));
  Pose3PointVelocityContactFactor factor(
      0, 1, 2, 3, Point3(0.2, -0.1, 0.4), Vector3(0.03, -0.02, 0.01),
      Vector3(0.2, -0.1, 0.4), noiseModel::Unit::Create(6));

  Matrix actual_H_pose, actual_H_velocity, actual_H_foothold, actual_H_bias;
  factor.evaluateError(pose, velocity, foothold, bias, actual_H_pose,
                       actual_H_velocity, actual_H_foothold, actual_H_bias);
  const auto error = [&](const Pose3& x, const Vector3& v, const Point3& p,
                         const imuBias::ConstantBias& b) {
    return factor.evaluateError(x, v, p, b, {}, {}, {}, {});
  };

  EXPECT(assert_equal(numericalDerivative41<Vector, Pose3, Vector3, Point3,
                                            imuBias::ConstantBias>(
                          error, pose, velocity, foothold, bias, 1e-6),
                      actual_H_pose, 1e-6));
  EXPECT(assert_equal(numericalDerivative42<Vector, Pose3, Vector3, Point3,
                                            imuBias::ConstantBias>(
                          error, pose, velocity, foothold, bias, 1e-6),
                      actual_H_velocity, 1e-6));
  EXPECT(assert_equal(numericalDerivative43<Vector, Pose3, Vector3, Point3,
                                            imuBias::ConstantBias>(
                          error, pose, velocity, foothold, bias, 1e-6),
                      actual_H_foothold, 1e-6));
  EXPECT(assert_equal(numericalDerivative44<Vector, Pose3, Vector3, Point3,
                                            imuBias::ConstantBias>(
                          error, pose, velocity, foothold, bias, 1e-6),
                      actual_H_bias, 1e-6));
}

/* ************************************************************************* */
TEST(LeggedEstimatorFactors, PointHeightFactorJacobian) {
  const Point3 foothold(1.2, 0.8, -0.5);
  PointHeightFactor factor(0, -0.35, noiseModel::Unit::Create(1));

  Matrix actual_H_foothold;
  factor.evaluateError(foothold, actual_H_foothold);
  const Matrix expected_H_foothold = numericalDerivative11<Vector, Point3>(
      [&](const Point3& p) { return factor.evaluateError(p, {}); }, foothold,
      1e-6);

  EXPECT(assert_equal(expected_H_foothold, actual_H_foothold, 1e-6));
}

/* ************************************************************************* */
TEST(LeggedEstimatorFactors, PointVelocityFactorsRejectInvalidNoiseDimensions) {
  const std::array<SharedNoiseModel, 2> invalidModels{
      nullptr, noiseModel::Unit::Create(3)};
  for (const SharedNoiseModel& model : invalidModels) {
    CHECK_EXCEPTION(
        ExtendedPosePointVelocityContactFactor(
            0, 2, Point3::Zero(), Vector3::Zero(), Vector3::Zero(), model),
        std::invalid_argument);
    CHECK_EXCEPTION(
        NavStatePointVelocityContactFactor(
            0, 1, 2, Point3::Zero(), Vector3::Zero(), Vector3::Zero(), model),
        std::invalid_argument);
    CHECK_EXCEPTION(Pose3PointVelocityContactFactor(0, 1, 2, 3, Point3::Zero(),
                                                    Vector3::Zero(),
                                                    Vector3::Zero(), model),
                    std::invalid_argument);
  }
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
