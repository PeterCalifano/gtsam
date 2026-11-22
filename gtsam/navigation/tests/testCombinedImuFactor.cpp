/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file    testCombinedImuFactor.cpp
 * @brief   Unit test for Lupton-style combined IMU factor
 * @author  Luca Carlone
 * @author  Frank Dellaert
 * @author  Richard Roberts
 * @author  Stephen Williams
 * @author  Varun Agrawal
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/MatrixConstants.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/VectorConstants.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/ImuBias.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/ScenarioRunner.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/factorTesting.h>

#include <array>
#include <cmath>
#include <limits>
#include <list>
#include <optional>
#include <type_traits>

#include "imuFactorTesting.h"

namespace combined {
// Create default parameters with Z-down and above noise parameters
static std::shared_ptr<PreintegratedCombinedMeasurements::Params> Params(
    const Matrix3& biasAccCovariance = Matrix3::Zero(),
    const Matrix3& biasOmegaCovariance = Matrix3::Zero()) {
  auto p = PreintegratedCombinedMeasurements::Params::MakeSharedD(kGravity);
  p->gyroscopeCovariance = kGyroSigma * kGyroSigma * I_3x3;
  p->accelerometerCovariance = kAccelSigma * kAccelSigma * I_3x3;
  p->integrationCovariance = 0.0001 * I_3x3;
  p->biasAccCovariance = biasAccCovariance;
  p->biasOmegaCovariance = biasOmegaCovariance;
  return p;
}
}  // namespace combined

/* ************************************************************************* */
TEST_PIM_WITH_COMBINED_BACKEND(CombinedImuFactor, PreintegratedMeasurements) {
  // Linearization point
  Bias bias(
      Vector3(0, 0, 0),
      Vector3(
          0, 0,
          0));  ///< Current estimate of acceleration and angular rate biases

  // Measurements
  Vector3 measuredAcc(0.1, 0.0, 0.0);
  Vector3 measuredOmega(M_PI / 100.0, 0.0, 0.0);
  double deltaT = 0.5;
  double tol = 1e-6;

  auto p = combined::Params();

  // Actual preintegrated values
  PIM expected1(p, bias);
  expected1.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  CombinedPIM actual1(p, bias);

  actual1.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  EXPECT(assert_equal(Vector(expected1.deltaPij()), actual1.deltaPij(), tol));
  EXPECT(assert_equal(Vector(expected1.deltaVij()), actual1.deltaVij(), tol));
  EXPECT(assert_equal(expected1.deltaRij(), actual1.deltaRij(), tol));
  DOUBLES_EQUAL(expected1.deltaTij(), actual1.deltaTij(), tol);
}

/* ************************************************************************* */
TEST_PIM(CombinedImuFactor, ReturnedDynamicsReconstructCovariance) {
  const auto params = combined::Params(1e-8 * I_3x3, 1e-10 * I_3x3);
  const Bias bias(Vector3(0.01, -0.02, 0.03),
                  Vector3(-0.001, 0.002, -0.003));

  // A nonzero prior exercises both propagation and the added process noise.
  const Eigen::Matrix<double, 15, 15> initialCovariance =
      1e-4 * Eigen::Matrix<double, 15, 15>::Identity();
  CombinedPIM observed(params, bias, initialCovariance);
  CombinedPIM standard(params, bias, initialCovariance);
  Eigen::Matrix<double, 15, 15> propagated = initialCovariance;

  const std::array<Vector3, 3> accelerations{
      Vector3(0.1, -0.2, 9.7), Vector3(0.2, -0.1, 9.8),
      Vector3(-0.1, 0.3, 9.6)};
  const std::array<Vector3, 3> angularVelocities{
      Vector3(0.02, -0.03, 0.01), Vector3(-0.01, 0.04, 0.03),
      Vector3(0.03, 0.02, -0.02)};
  const std::array<double, 3> intervals{0.004, 0.007, 0.005};

  for (size_t index = 0; index < intervals.size(); ++index) {
    const auto step = observed.integrateMeasurementWithDynamics(
        accelerations[index], angularVelocities[index], intervals[index]);
    standard.integrateMeasurement(accelerations[index],
                                  angularVelocities[index], intervals[index]);
    propagated = step.transition * propagated * step.transition.transpose() +
                 step.processCovariance;

    EXPECT(assert_equal(propagated, observed.preintMeasCov(), 1e-12));
    EXPECT(assert_equal(standard.preintMeasCov(), observed.preintMeasCov(),
                        1e-12));
    EXPECT(assert_equal(standard.deltaRij(), observed.deltaRij(), 1e-12));
    EXPECT(assert_equal(standard.deltaPij(), observed.deltaPij(), 1e-12));
    EXPECT(assert_equal(standard.deltaVij(), observed.deltaVij(), 1e-12));
    DOUBLES_EQUAL(standard.deltaTij(), observed.deltaTij(), 1e-12);

    EXPECT(step.gyroSampleCovariance.isApprox(
        params->gyroscopeCovariance / intervals[index], 1e-12));
    // Gyro noise never enters the bias rows. It reaches position and velocity
    // only for Galilean preintegration, as no sensor pose is set here.
    EXPECT(step.gyroNoiseMap.bottomRows(6).isZero(1e-12));
    if constexpr (!std::is_same_v<CombinedPIM,
                                  PreintegratedCombinedMeasurementsG>) {
      EXPECT(step.gyroNoiseMap.middleRows(3, 6).isZero(1e-12));
    }
  }
}

/* ************************************************************************* */
TEST_PIM(CombinedImuFactor, ReturnedGyroNoiseReconstructsProcessCovariance) {
  // Isolate gyro noise so the returned map must explain the full noise term.
  const auto params = combined::Params();
  params->accelerometerCovariance.setZero();
  params->integrationCovariance.setZero();
  CombinedPIM observed(params);
  const Vector3 acceleration(0.3, -0.2, 9.7);
  const Vector3 angularVelocity(0.2, -0.1, 0.4);

  for (const double dt : {0.004, 0.007, 0.005}) {
    const auto step = observed.integrateMeasurementWithDynamics(
        acceleration, angularVelocity, dt);
    const Matrix expected = step.gyroNoiseMap * step.gyroSampleCovariance *
                            step.gyroNoiseMap.transpose();

    EXPECT(assert_equal(expected, step.processCovariance, 1e-12));
    EXPECT(expected.trace() > 0.0);
  }
}

/* ************************************************************************* */
namespace sample_noise {

using Matrix15 = Eigen::Matrix<double, 15, 15>;
using ManifoldCombinedPIM =
    PreintegratedCombinedMeasurementsT<ManifoldPreintegration>;

/// Anisotropic densities with bias random walk and an integration floor.
std::shared_ptr<PreintegrationCombinedParams> makeParams() {
  auto params = combined::Params(1e-8 * I_3x3, 1e-10 * I_3x3);
  params->accelerometerCovariance = Vector3(2e-4, 3e-4, 5e-4).asDiagonal();
  params->gyroscopeCovariance = Vector3(1e-5, 2e-5, 3e-5).asDiagonal();
  return params;
}

/// The per-call noise equivalent to the density call for one duration.
CombinedPreintegrationSampleNoise makeDensityNoise(
    const PreintegrationCombinedParams& params, double dt) {
  CombinedPreintegrationSampleNoise noise;
  noise.accelerometerSampleCovariance = params.accelerometerCovariance / dt;
  noise.gyroscopeSampleCovariance = params.gyroscopeCovariance / dt;
  return noise;
}

/// A full-rank increment covariance [rotation, position, velocity] with cross
/// terms, in rad^2, m^2 and (m/s)^2.
Matrix9 makeMotionCovariance() {
  Matrix9 root = Matrix9::Zero();
  for (int row = 0; row < 9; ++row) {
    for (int column = 0; column <= row; ++column) {
      root(row, column) = (row == column ? 1.0 : 0.3) * (1 + row - column);
    }
  }
  const Vector9 scale =
      (Vector9() << 1e-3, 1e-3, 1e-3, 2e-4, 2e-4, 4e-4, 8e-2, 8e-2, 8e-2)
          .finished();
  root = scale.asDiagonal() * root;
  return root * root.transpose();
}

const std::array<Vector3, 3> kAccelerations{
    Vector3(0.1, -0.2, 9.7), Vector3(0.2, -0.1, 9.8), Vector3(-0.1, 0.3, 9.6)};
const std::array<Vector3, 3> kAngularVelocities{
    Vector3(0.4, -0.3, 1.0), Vector3(-0.1, 0.4, 0.3), Vector3(0.3, 0.2, -0.2)};
const std::array<double, 3> kIntervals{0.0125, 0.044, 0.0025};

// Density covariances divided by dt reproduce the density call bit for bit in
// both integration calls, also with a displaced and rotated sensor.
TEST_PIM(CombinedImuFactor, SampleNoiseMatchesDensityCall) {
  for (const bool hasSensorPose : {false, true}) {
    const auto params = makeParams();
    if (hasSensorPose) {
      params->body_P_sensor =
          Pose3(Rot3::RzRyRx(0.1, -0.2, 0.3), Point3(0.05, -0.02, 0.1));
    }
    const Bias bias(Vector3(0.01, -0.02, 0.03), Vector3(-0.001, 0.002, 0.0));
    const Matrix15 prior = 1e-4 * Matrix15::Identity();
    CombinedPIM density(params, bias, prior), perCall(params, bias, prior);
    CombinedPIM densityDynamics(params, bias, prior);
    CombinedPIM perCallDynamics(params, bias, prior);

    for (size_t index = 0; index < kIntervals.size(); ++index) {
      const Vector3& acceleration = kAccelerations[index];
      const Vector3& angularVelocity = kAngularVelocities[index];
      const double dt = kIntervals[index];
      const auto noise = makeDensityNoise(*params, dt);
      density.integrateMeasurement(acceleration, angularVelocity, dt);
      perCall.integrateMeasurement(acceleration, angularVelocity, dt, noise);
      const auto expected = densityDynamics.integrateMeasurementWithDynamics(
          acceleration, angularVelocity, dt);
      const auto actual = perCallDynamics.integrateMeasurementWithDynamics(
          acceleration, angularVelocity, dt, noise);

      EXPECT(perCall.preintMeasCov() == density.preintMeasCov());
      EXPECT(perCall.deltaRij().matrix() == density.deltaRij().matrix());
      EXPECT(perCall.deltaPij() == density.deltaPij());
      EXPECT(perCall.deltaVij() == density.deltaVij());
      EXPECT(perCallDynamics.preintMeasCov() ==
             densityDynamics.preintMeasCov());
      EXPECT(actual.transition == expected.transition);
      EXPECT(actual.processCovariance == expected.processCovariance);
      EXPECT(actual.gyroNoiseMap == expected.gyroNoiseMap);
      EXPECT(actual.gyroSampleCovariance == expected.gyroSampleCovariance);
    }
  }
}

// One reading held for h with sample covariance S gives h^4/4 S, h^3/2 S and
// h^2 S in position, position-velocity and velocity, and h^2 S_g in rotation.
TEST(CombinedImuFactor, HeldSampleCovariance) {
  const auto params = combined::Params(Matrix3::Zero(), Matrix3::Zero());
  params->integrationCovariance.setZero();
  const double h = 0.044;
  CombinedPreintegrationSampleNoise noise;
  noise.accelerometerSampleCovariance =
      Vector3(1.1e-3, 1.2e-3, 2.2e-3).asDiagonal();
  noise.gyroscopeSampleCovariance =
      Vector3(3.6e-5, 1.2e-4, 4.4e-5).asDiagonal();
  ManifoldCombinedPIM pim(params);
  pim.integrateMeasurement(Vector3(0.3, -0.2, 9.7), Vector3::Zero(), h, noise);

  const Matrix15 covariance = pim.preintMeasCov();
  const Matrix3& accelerometer = noise.accelerometerSampleCovariance;
  EXPECT((covariance.block<3, 3>(0, 0).isApprox(
      h * h * noise.gyroscopeSampleCovariance, 1e-14)));
  EXPECT((covariance.block<3, 3>(3, 3).isApprox(
      std::pow(h, 4) / 4 * accelerometer, 1e-14)));
  EXPECT((covariance.block<3, 3>(3, 6).isApprox(
      std::pow(h, 3) / 2 * accelerometer, 1e-14)));
  EXPECT((covariance.block<3, 3>(6, 6).isApprox(h * h * accelerometer, 1e-14)));
  EXPECT((covariance.block<3, 6>(0, 3).isZero(0.0)));
  EXPECT((covariance.block<6, 15>(9, 0).isZero(0.0)));
}

// The motion covariance enters the end-of-step Manifold chart. The map is
// derived independently from perturbing the integrated NavState: translation
// errors in the start-of-step axes, rotation as a right end perturbation.
TEST(CombinedImuFactor, MotionCovarianceInEndOfStepChart) {
  const auto params = combined::Params(Matrix3::Zero(), Matrix3::Zero());
  params->integrationCovariance.setZero();
  const CombinedPreintegrationSampleNoise zeroNoise{
      Matrix3::Zero(), Matrix3::Zero(), std::nullopt};
  auto noise = zeroNoise;
  noise.motionCovariance = makeMotionCovariance();

  // Rotate first with zero noise so the start-of-step axes differ from i.
  ManifoldCombinedPIM pim(params);
  pim.integrateMeasurement(Vector3(0.1, 0.2, 9.8), Vector3(0.5, -0.4, 0.8), 0.1,
                           zeroNoise);
  const ManifoldCombinedPIM before = pim;
  const Rot3 startAttitude = pim.deltaRij();
  pim.integrateMeasurement(kAccelerations[0], kAngularVelocities[0],
                           kIntervals[1], noise);

  const NavState nominal = pim.deltaXij();
  constexpr double kStep = 1e-3;
  Matrix9 motionMap;
  for (int column = 0; column < 9; ++column) {
    const Vector9 error = kStep * Vector9::Unit(column);
    const NavState perturbed(
        nominal.attitude() * Rot3::Expmap(error.head<3>()),
        nominal.position() + startAttitude.rotate(error.segment<3>(3)),
        nominal.velocity() + startAttitude.rotate(error.tail<3>()));
    motionMap.col(column) =
        internal::navStateComponentWiseLocalCoordinates(nominal, perturbed) /
        kStep;
  }
  const Matrix9 expected =
      motionMap * *noise.motionCovariance * motionMap.transpose();
  EXPECT((Matrix9(pim.preintMeasCov().topLeftCorner<9, 9>())
              .isApprox(expected, 1e-9)));
  EXPECT(pim.preintMeasCov().bottomRows<6>().isZero(0.0));

  // The dense step carries the same term in processCovariance only; the gyro
  // moments are those of the sensor sample.
  ManifoldCombinedPIM dynamics = before, density = before;
  const auto step = dynamics.integrateMeasurementWithDynamics(
      kAccelerations[0], kAngularVelocities[0], kIntervals[1], noise);
  const auto densityStep = density.integrateMeasurementWithDynamics(
      kAccelerations[0], kAngularVelocities[0], kIntervals[1]);
  EXPECT(dynamics.preintMeasCov() == pim.preintMeasCov());
  EXPECT((Matrix9(step.processCovariance.topLeftCorner<9, 9>())
              .isApprox(expected, 1e-9)));
  EXPECT(step.transition == densityStep.transition);
  EXPECT(step.gyroNoiseMap == densityStep.gyroNoiseMap);
  EXPECT(step.gyroSampleCovariance.isZero(0.0));
}

// With sensor and motion noise, F P F' + Q of the returned steps reproduces
// the sparse covariance propagation.
TEST(CombinedImuFactor, MotionCovarianceDynamicsReconstructCovariance) {
  const auto params = makeParams();
  const Bias bias(Vector3(0.01, -0.02, 0.03), Vector3(-0.001, 0.002, 0.0));
  const Matrix15 prior = 1e-4 * Matrix15::Identity();
  ManifoldCombinedPIM sparse(params, bias, prior), dense(params, bias, prior);
  Matrix15 propagated = prior;
  for (size_t index = 0; index < kIntervals.size(); ++index) {
    auto noise = makeDensityNoise(*params, 2.5e-3);
    noise.motionCovariance = makeMotionCovariance();
    sparse.integrateMeasurement(kAccelerations[index],
                                kAngularVelocities[index], kIntervals[index],
                                noise);
    const auto step = dense.integrateMeasurementWithDynamics(
        kAccelerations[index], kAngularVelocities[index], kIntervals[index],
        noise);
    propagated = step.transition * propagated * step.transition.transpose() +
                 step.processCovariance;
    EXPECT(assert_equal(propagated, sparse.preintMeasCov(), 1e-12));
    EXPECT(dense.preintMeasCov() == sparse.preintMeasCov());
  }
}

// Invalid noise, a nonpositive duration and a motion covariance on another
// backend all throw before the PIM changes; a singular PSD input is accepted.
TEST(CombinedImuFactor, InvalidSampleNoiseLeavesPimUnchanged) {
  const auto params = makeParams();
  ManifoldCombinedPIM pim(params);
  pim.integrateMeasurement(kAccelerations[0], kAngularVelocities[0],
                           kIntervals[0]);
  const ManifoldCombinedPIM original = pim;
  const auto unchanged = [&]() {
    return pim.preintMeasCov() == original.preintMeasCov() &&
           pim.deltaTij() == original.deltaTij() &&
           pim.deltaRij().matrix() == original.deltaRij().matrix() &&
           pim.deltaPij() == original.deltaPij() &&
           pim.deltaVij() == original.deltaVij();
  };

  const auto valid = makeDensityNoise(*params, 2.5e-3);
  std::array<CombinedPreintegrationSampleNoise, 3> invalid{valid, valid, valid};
  invalid[0].accelerometerSampleCovariance(1, 1) =
      std::numeric_limits<double>::quiet_NaN();
  invalid[1].gyroscopeSampleCovariance(0, 2) = 1e-6;
  invalid[2].motionCovariance = makeMotionCovariance();
  (*invalid[2].motionCovariance)(4, 4) = -1e-6;
  for (const auto& noise : invalid) {
    CHECK_EXCEPTION(
        pim.integrateMeasurement(kAccelerations[1], kAngularVelocities[1],
                                 kIntervals[1], noise),
        std::invalid_argument);
    CHECK_EXCEPTION(
        pim.integrateMeasurementWithDynamics(
            kAccelerations[1], kAngularVelocities[1], kIntervals[1], noise),
        std::invalid_argument);
    EXPECT(unchanged());
  }
  CHECK_EXCEPTION(pim.integrateMeasurement(kAccelerations[1],
                                           kAngularVelocities[1], 0.0, valid),
                  std::runtime_error);
  EXPECT(unchanged());

  auto withMotion = valid;
  withMotion.motionCovariance = makeMotionCovariance();
  PreintegratedCombinedMeasurementsT<TangentPreintegration> tangent(params);
  CHECK_EXCEPTION(
      tangent.integrateMeasurement(kAccelerations[1], kAngularVelocities[1],
                                   kIntervals[1], withMotion),
      std::invalid_argument);
  EXPECT(tangent.deltaTij() == 0.0);

  // A held-reading block [h^4/4, h^3/2; h^3/2, h^2] S has rank three.
  const double h = 0.044;
  Matrix9 singular = Matrix9::Zero();
  singular.block<3, 3>(3, 3) = std::pow(h, 4) / 4 * I_3x3;
  singular.block<3, 3>(3, 6) = singular.block<3, 3>(6, 3) =
      std::pow(h, 3) / 2 * I_3x3;
  singular.block<3, 3>(6, 6) = h * h * I_3x3;
  withMotion.motionCovariance = 1e-2 * singular;
  pim.integrateMeasurement(kAccelerations[1], kAngularVelocities[1], h,
                           withMotion);
  EXPECT(!unchanged());
}

using TangentCombinedPIM =
    PreintegratedCombinedMeasurementsT<TangentPreintegration>;

/// Error of a perturbed PIM in the Manifold covariance chart.
Vector9 chartError(const ManifoldCombinedPIM& nominal,
                   const ManifoldCombinedPIM& perturbed) {
  return internal::navStateComponentWiseLocalCoordinates(nominal.deltaXij(),
                                                         perturbed.deltaXij());
}

/// Error of a perturbed PIM in the Tangent covariance chart.
Vector9 chartError(const TangentCombinedPIM& nominal,
                   const TangentCombinedPIM& perturbed) {
  return perturbed.preintegrated() - nominal.preintegrated();
}

/// A full covariance R diag(sigma^2) R' with the given principal axes.
Matrix3 makeFullCovariance(const Rot3& axes, const Vector3& sigmas) {
  return axes.matrix() * sigmas.cwiseAbs2().asDiagonal() *
         axes.matrix().transpose();
}

/**
 * Check the per-call covariance against a first-order oracle that does not use
 * the PIM's noise maps. With zero densities, bias random walk, integration
 * noise and prior, P = sum_k J_k S_k J_k', where J_k is the central-difference
 * derivative of the final chart error with respect to the sensor reading of
 * step k and S_k is its sample covariance. The steps rotate, the sample
 * covariances are full and differ per step, and the sensor can be displaced
 * and rotated, so the oracle also checks that S_k is taken in sensor axes.
 */
template <class PIMType>
void checkSampleNoiseAgainstReadingJacobians(TestResult& result_,
                                             const std::string& name_) {
  const std::array<Vector3, 3> accelerations{Vector3(0.6, -0.4, 9.5),
                                             Vector3(-0.3, 0.8, 9.9),
                                             Vector3(0.2, 0.1, 10.2)};
  const std::array<Vector3, 3> angularVelocities{Vector3(1.2, -0.7, 2.0),
                                                 Vector3(-0.9, 1.5, 0.6),
                                                 Vector3(0.4, 0.8, -1.7)};
  const std::array<double, 3> intervals{0.05, 0.0125, 0.03};
  std::array<CombinedPreintegrationSampleNoise, 3> noises;
  for (size_t step = 0; step < noises.size(); ++step) {
    const double scale = 1.0 + step;
    noises[step].accelerometerSampleCovariance =
        makeFullCovariance(Rot3::RzRyRx(0.3 * scale, -0.5, 0.2),
                           scale * Vector3(3e-2, 5e-2, 9e-2));
    noises[step].gyroscopeSampleCovariance =
        makeFullCovariance(Rot3::RzRyRx(-0.4, 0.6 * scale, 0.1),
                           scale * Vector3(2e-3, 4e-3, 7e-3));
  }

  for (const bool hasSensorPose : {false, true}) {
    const auto params = combined::Params(Matrix3::Zero(), Matrix3::Zero());
    params->integrationCovariance.setZero();
    if (hasSensorPose) {
      params->body_P_sensor =
          Pose3(Rot3::RzRyRx(0.4, -0.3, 0.7), Point3(0.08, -0.05, 0.12));
    }
    const Bias bias(Vector3(0.05, -0.02, 0.03), Vector3(0.01, -0.02, 0.015));

    // Integrate the mean with one reading component of one step shifted.
    const auto integrate = [&](size_t perturbedStep, int component,
                               double shift) {
      PIMType pim(params, bias);
      for (size_t step = 0; step < intervals.size(); ++step) {
        Vector6 reading;
        reading << accelerations[step], angularVelocities[step];
        if (step == perturbedStep) reading(component) += shift;
        pim.integrateMeasurement(reading.head<3>(), reading.tail<3>(),
                                 intervals[step]);
      }
      return pim;
    };
    const PIMType nominal = integrate(intervals.size(), 0, 0.0);

    Matrix9 expected = Matrix9::Zero();
    for (size_t step = 0; step < intervals.size(); ++step) {
      Eigen::Matrix<double, 9, 6> readingJacobian;
      for (int component = 0; component < 6; ++component) {
        const double shift = component < 3 ? 1e-4 : 1e-5;
        readingJacobian.col(component) =
            (chartError(nominal, integrate(step, component, shift)) -
             chartError(nominal, integrate(step, component, -shift))) /
            (2 * shift);
      }
      Matrix6 sampleCovariance = Matrix6::Zero();
      sampleCovariance.topLeftCorner<3, 3>() =
          noises[step].accelerometerSampleCovariance;
      sampleCovariance.bottomRightCorner<3, 3>() =
          noises[step].gyroscopeSampleCovariance;
      expected +=
          readingJacobian * sampleCovariance * readingJacobian.transpose();
    }

    PIMType actual(params, bias);
    for (size_t step = 0; step < intervals.size(); ++step) {
      actual.integrateMeasurement(accelerations[step], angularVelocities[step],
                                  intervals[step], noises[step]);
    }
    const Matrix9 navigation =
        actual.preintMeasCov().template topLeftCorner<9, 9>();
    const double relativeError =
        (navigation - expected).norm() / expected.norm();
    EXPECT_DOUBLES_EQUAL(0.0, relativeError, 1e-6);
    EXPECT(actual.preintMeasCov().template bottomRows<6>().isZero(0.0));
  }
}

// The per-call covariance matches the first-order effect of the sensor
// readings on the preintegrated delta, in both established covariance charts.
TEST(CombinedImuFactor, SampleNoiseMatchesReadingJacobians) {
  checkSampleNoiseAgainstReadingJacobians<ManifoldCombinedPIM>(result_, name_);
  checkSampleNoiseAgainstReadingJacobians<TangentCombinedPIM>(result_, name_);
}

// Validation tolerates rounding asymmetry, and the update then uses the
// symmetric part, so the result equals that of the explicitly symmetric input.
TEST(CombinedImuFactor, SampleNoiseUsesSymmetricPart) {
  const auto params = makeParams();
  CombinedPreintegrationSampleNoise symmetric;
  symmetric.accelerometerSampleCovariance = makeFullCovariance(
      Rot3::RzRyRx(0.2, 0.4, -0.1), Vector3(3e-2, 4e-2, 6e-2));
  symmetric.gyroscopeSampleCovariance = makeFullCovariance(
      Rot3::RzRyRx(-0.3, 0.1, 0.5), Vector3(2e-3, 3e-3, 5e-3));
  auto rounded = symmetric;
  const double offset =
      1e-13 * rounded.accelerometerSampleCovariance.cwiseAbs().maxCoeff();
  rounded.accelerometerSampleCovariance(0, 1) += offset;
  rounded.accelerometerSampleCovariance(1, 0) -= offset;
  const Matrix3 symmetricPart =
      0.5 * (rounded.accelerometerSampleCovariance +
             rounded.accelerometerSampleCovariance.transpose());
  auto expectedNoise = symmetric;
  expectedNoise.accelerometerSampleCovariance = symmetricPart;

  ManifoldCombinedPIM actual(params), expected(params);
  actual.integrateMeasurement(kAccelerations[0], kAngularVelocities[0],
                              kIntervals[0], rounded);
  expected.integrateMeasurement(kAccelerations[0], kAngularVelocities[0],
                                kIntervals[0], expectedNoise);
  EXPECT(actual.preintMeasCov() == expected.preintMeasCov());
  EXPECT(actual.preintMeasCov() == Matrix(actual.preintMeasCov().transpose()));
}

}  // namespace sample_noise

/* ************************************************************************* */
// Checks equivalence between matching three-way and combined PIM backends.
TEST_PIM_WITH_COMBINED_BACKEND(CombinedImuFactor, ErrorWithBiases) {
  Bias bias(Vector3(0.2, 0, 0), Vector3(0, 0, 0.3));     // Biases (acc, rot)
  Bias bias2(Vector3(0.2, 0.2, 0), Vector3(1, 0, 0.3));  // Biases (acc, rot)
  Pose3 x1(Rot3::Expmap(Vector3(0, 0, M_PI / 4.0)), Point3(5.0, 1.0, -50.0));
  Vector3 v1(0.5, 0.0, 0.0);
  Pose3 x2(Rot3::Expmap(Vector3(0, 0, M_PI / 4.0 + M_PI / 10.0)),
           Point3(5.5, 1.0, -50.0));
  Vector3 v2(0.5, 0.0, 0.0);

  auto p = combined::Params();
  p->omegaCoriolis = Vector3(0, 0.1, 0.1);
  PIM pim(p, Bias(Vector3(0.2, 0.0, 0.0), Vector3(0.0, 0.0, 0.0)));

  // Measurements
  Vector3 measuredOmega{0, 0, M_PI / 10.0 + 0.3};
  Vector3 measuredAcc =
      x1.rotation().unrotate(-p->n_gravity) + Vector3(0.2, 0.0, 0.0);
  double deltaT = 1.0;
  double tol = 1e-6;

  pim.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  CombinedPIM combined_pim(
      p, Bias(Vector3(0.2, 0.0, 0.0), Vector3(0.0, 0.0, 0.0)));

  combined_pim.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  // Create factor
  ImuFactorT<PIM> imuFactor(X(1), V(1), X(2), V(2), B(1), pim);

  CombinedImuFactorT<CombinedPIM> combinedFactor(X(1), V(1), X(2), V(2), B(1),
                                                 B(2), combined_pim);

  Vector errorExpected = imuFactor.evaluateError(x1, v1, x2, v2, bias);
  Vector errorActual =
      combinedFactor.evaluateError(x1, v1, x2, v2, bias, bias2);
  EXPECT(assert_equal(errorExpected, errorActual.head(9), tol));

  // Expected Jacobians
  Matrix H1e, H2e, H3e, H4e, H5e;
  (void)imuFactor.evaluateError(x1, v1, x2, v2, bias, H1e, H2e, H3e, H4e, H5e);

  // Actual Jacobians
  Matrix H1a, H2a, H3a, H4a, H5a, H6a;
  (void)combinedFactor.evaluateError(x1, v1, x2, v2, bias, bias2, H1a, H2a, H3a,
                                     H4a, H5a, H6a);

  EXPECT(assert_equal(H1e, H1a.topRows(9)));
  EXPECT(assert_equal(H2e, H2a.topRows(9)));
  EXPECT(assert_equal(H3e, H3a.topRows(9)));
  EXPECT(assert_equal(H4e, H4a.topRows(9)));
  EXPECT(assert_equal(H5e, H5a.topRows(9)));
}

/* ************************************************************************* */
// This test only works with tangent preintegration.
TEST(CombinedImuFactor, FirstOrderPreIntegratedMeasurements) {
  auto p = combined::Params();
  testing::SomeMeasurements measurements;

  auto preintegrated = [&](const Vector3& a, const Vector3& w) {
    PreintegratedImuMeasurementsT<TangentPreintegration> pim(p, Bias(a, w));
    testing::integrateMeasurements(measurements, &pim);
    return pim.preintegrated();
  };

  // Actual pre-integrated values
  PreintegratedCombinedMeasurementsT<TangentPreintegration> pim(p);
  testing::integrateMeasurements(measurements, &pim);

  EXPECT(assert_equal(numericalDerivative21<Vector9, Vector3, Vector3>(
                          preintegrated, Z_3x1, Z_3x1),
                      pim.preintegrated_H_biasAcc()));
  EXPECT(assert_equal(numericalDerivative22<Vector9, Vector3, Vector3>(
                          preintegrated, Z_3x1, Z_3x1),
                      pim.preintegrated_H_biasOmega(), 1e-3));
}

/* ************************************************************************* */
TEST_PIM(CombinedImuFactor, PredictPositionAndVelocity) {
  const Bias bias(Vector3(0, 0.1, 0), Vector3(0, 0.1, 0));  // Biases (acc, rot)

  auto p = combined::Params();

  // Measurements
  const Vector3 measuredOmega(0, 0.1, 0);  // M_PI/10.0+0.3;
  const Vector3 measuredAcc(0, 1.1, -kGravity);
  const double deltaT = 0.01;

  CombinedPIM pim(p, bias);

  for (int i = 0; i < 100; ++i)
    pim.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  // Create factor
  const noiseModel::Gaussian::shared_ptr combinedmodel =
      noiseModel::Gaussian::Covariance(pim.preintMeasCov());
  const CombinedImuFactorT<CombinedPIM> Combinedfactor(X(1), V(1), X(2), V(2),
                                                       B(1), B(2), pim);

  // Predict
  const NavState actual = pim.predict(NavState(), bias);
  const Pose3 expectedPose(Rot3(), Point3(0, 0.5, 0));
  const Vector3 expectedVelocity(0, 1, 0);
  EXPECT(assert_equal(expectedPose, actual.pose()));
  EXPECT(assert_equal(Vector(expectedVelocity), Vector(actual.velocity())));
}

/* ************************************************************************* */
TEST_PIM(CombinedImuFactor, PredictRotation) {
  const Bias bias(Vector3(0, 0, 0), Vector3(0, 0, 0));  // Biases (acc, rot)
  auto p = combined::Params();
  CombinedPIM pim(p, bias);
  const Vector3 measuredAcc = -kGravityAlongNavZDown;
  const Vector3 measuredOmega(0, 0, M_PI / 10.0);
  const double deltaT = 0.01;
  const double tol = 1e-4;
  for (int i = 0; i < 100; ++i)
    pim.integrateMeasurement(measuredAcc, measuredOmega, deltaT);
  const CombinedImuFactorT<CombinedPIM> Combinedfactor(X(1), V(1), X(2), V(2),
                                                       B(1), B(2), pim);

  // Predict
  const Pose3 x(Rot3::Ypr(0, 0, 0), Point3(0, 0, 0)), x2;
  const Vector3 v(0, 0, 0);
  const NavState actual = pim.predict(NavState(x, v), bias);
  const Pose3 expectedPose(Rot3::Ypr(M_PI / 10, 0, 0), Point3(0, 0, 0));
  EXPECT(assert_equal(expectedPose, actual.pose(), tol));
}

/* ************************************************************************* */
// Testing covariance to check if all the jacobians are accounted for.
TEST(CombinedImuFactor, CheckDefaultCovarianceRegression) {
  auto params = PreintegrationCombinedParams::MakeSharedU(9.81);

  params->setAccelerometerCovariance(pow(0.01, 2) * I_3x3);
  params->setGyroscopeCovariance(pow(1.75e-4, 2) * I_3x3);
  params->setIntegrationCovariance(pow(0.0, 2) * I_3x3);
  params->setOmegaCoriolis(Vector3::Zero());

  imuBias::ConstantBias currentBias;

  PreintegratedCombinedMeasurements actual(params, currentBias);

  // Measurements
  Vector3 measuredAcc(0.1577, -0.8251, 9.6111);
  Vector3 measuredOmega(-0.0210, 0.0311, 0.0145);
  double deltaT = 5;

  actual.integrateMeasurement(measuredAcc, measuredOmega, deltaT);

  Eigen::Matrix<double, 15, 15> expected{
      {1.53125e-07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 1.53125e-07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 1.53125e-07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0.003125, 0, 0, 0.00125, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0.003125, 0, 0, 0.00125, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0.003125, 0, 0, 0.00125, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0.00125, 0, 0, 0.0005, 0, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0.00125, 0, 0, 0.0005, 0, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0.00125, 0, 0, 0.0005, 0, 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 5., 0, 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5., 0, 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5., 0, 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5., 0, 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5., 0},
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5.}};

  // regression
  EXPECT(assert_equal(expected, actual.preintMeasCov()));
}

/* ************************************************************************* */
// Test that the covariance values for the ImuFactor and the CombinedImuFactor
// (top-left 9x9) are the same
TEST_PIM_WITH_COMBINED_BACKEND(CombinedImuFactor, SameCovariance) {
  // IMU measurements and time delta
  Vector3 accMeas(0.1577, -0.8251, 9.6111);
  Vector3 omegaMeas(-0.0210, 0.0311, 0.0145);
  double deltaT = 0.01;

  // Assume zero bias
  imuBias::ConstantBias currentBias;

  // Define params for ImuFactor
  auto params = PreintegrationParams::MakeSharedU();
  params->setAccelerometerCovariance(pow(0.01, 2) * I_3x3);
  params->setGyroscopeCovariance(pow(1.75e-4, 2) * I_3x3);
  params->setIntegrationCovariance(pow(0, 2) * I_3x3);
  params->setOmegaCoriolis(Vector3::Zero());

  // The IMU preintegration object for ImuFactor
  PIM pim(params, currentBias);
  pim.integrateMeasurement(accMeas, omegaMeas, deltaT);

  // Define params for CombinedImuFactor
  auto combined_params = PreintegrationCombinedParams::MakeSharedU();
  combined_params->setAccelerometerCovariance(pow(0.01, 2) * I_3x3);
  combined_params->setGyroscopeCovariance(pow(1.75e-4, 2) * I_3x3);
  // Set bias integration covariance explicitly to zero
  combined_params->setIntegrationCovariance(Z_3x3);
  combined_params->setOmegaCoriolis(Z_3x1);

  // The IMU preintegration object for CombinedImuFactor
  CombinedPIM cpim(combined_params, currentBias);
  cpim.integrateMeasurement(accMeas, omegaMeas, deltaT);

  // Assert if the noise covariance
  EXPECT(assert_equal<Matrix9>(
      pim.preintMeasCov(), Matrix9(cpim.preintMeasCov().block(0, 0, 9, 9))));
}

/* ************************************************************************* */
// Runners currently still use the integration type based on the compile flag.
// So we use the default CombinedScenarioRunner below.
TEST(CombinedImuFactor, Accelerating) {
  const double a = 0.2, v = 50;

  // Set up body pointing towards y axis, and start at 10,20,0 with velocity
  // going in X The body itself has Z axis pointing down
  const Rot3 nRb(Point3(0, 1, 0), Point3(1, 0, 0), Point3(0, 0, -1));
  const Point3 initial_position(10, 20, 0);
  const Vector3 initial_velocity(v, 0, 0);

  const AcceleratingScenario scenario(nRb, initial_position, initial_velocity,
                                      Vector3(a, 0, 0));

  const double T = 3.0;  // seconds

  CombinedScenarioRunner runner(scenario, combined::Params(), T / 10);

  PreintegratedCombinedMeasurements pim = runner.integrate(T);
  EXPECT(assert_equal(scenario.pose(T), runner.predict(pim).pose(), 1e-9));

  auto estimatedCov = runner.estimateCovariance(T, 100);
  Eigen::Matrix<double, 15, 15> expected = pim.preintMeasCov();
  EXPECT(assert_equal(estimatedCov, expected, 0.1));
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
/* ************************************************************************* */
