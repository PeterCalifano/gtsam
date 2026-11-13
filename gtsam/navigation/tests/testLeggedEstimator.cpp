/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 *
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file testLeggedEstimator.cpp
 * @date February 2026
 * @brief Unit tests for the legged estimator variants.
 * @author Pietro Califano (joint contact and covariance regressions)
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/MatrixConstants.h>
#include <gtsam/base/TestableAssertions.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/navigation/LeggedEstimator.h>
#include <gtsam/navigation/LeggedEstimatorFactors.h>
#include <gtsam/navigation/NavStateImuEKF.h>

#include <Eigen/Cholesky>
#include <limits>

using namespace gtsam;

namespace {

class ExposedLeggedInvariantEKF : public LeggedInvariantEKF {
 public:
  using LeggedInvariantEKF::applyCorrelatedContactUpdate;
  using LeggedInvariantEKF::LeggedInvariantEKF;
  using LeggedInvariantEKF::marginalizeFoot;
};

constexpr size_t kTwoFeet = 2;
constexpr int kExtendedPoseDim = 3 + 3 * static_cast<int>(2 + kTwoFeet);

LeggedEstimatorParams makeParams(bool marginalizeLeavingFoot = false) {
  LeggedEstimatorParams params;
  params.preintegrationParams = PreintegrationParams::MakeSharedU(9.81);
  params.preintegrationParams->setGyroscopeCovariance(I_3x3 * 1e-4);
  params.preintegrationParams->setIntegrationCovariance(I_3x3 * 1e-4);
  params.preintegrationParams->setAccelerometerCovariance(I_3x3 * 1e-4);
  params.body_P_imu = Pose3::Identity();
  params.footholdInitSigma = 0.25;
  params.footholdProcessSigma = 1e-3;
  params.contactCovariance = I_3x3 * (0.05 * 0.05);
  params.heightPriorSigma = 0.05;
  params.useFullContactInitialization = false;
  params.marginalizeLeavingFoot = marginalizeLeavingFoot;
  return params;
}

Matrix initialCovariance(size_t numFeet) {
  const int dim = 9 + 3 * static_cast<int>(numFeet);
  Matrix P = Matrix::Identity(dim, dim) * 0.1;
  P.block(0, 0, 3, 3) = I_3x3 * 0.01;
  P.block(3, 3, 3, 3) = I_3x3 * 0.04;
  P.block(6, 6, 3, 3) = I_3x3 * 0.04;
  for (size_t foot = 0; foot < numFeet; ++foot) {
    P.block(9 + 3 * static_cast<int>(foot), 9 + 3 * static_cast<int>(foot), 3,
            3) = I_3x3 * 0.09;
  }
  return P;
}

Matrix denseCorrelatedCovariance(size_t numFeet) {
  const int dim = 9 + 3 * static_cast<int>(numFeet);
  Matrix basis(dim, dim);
  for (int row = 0; row < dim; ++row) {
    for (int col = 0; col < dim; ++col) {
      basis(row, col) =
          0.13 * (row + 1) + 0.07 * (col + 1) + 0.01 * (row + 1) * (col + 1);
    }
  }
  return basis * basis.transpose() + Matrix::Identity(dim, dim);
}

Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>
makeMarginalizationPermutation(size_t numFeet, size_t marginalizedFoot) {
  const int dim = 9 + 3 * static_cast<int>(numFeet);
  const int footStart = 9 + 3 * static_cast<int>(marginalizedFoot);
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permutation(dim);
  Eigen::VectorXi indices(dim);
  int next = 0;
  for (int index = 0; index < dim; ++index) {
    if (index < footStart || index >= footStart + 3) {
      indices(next++) = index;
    }
  }
  for (int index = footStart; index < footStart + 3; ++index) {
    indices(next++) = index;
  }
  for (int row = 0; row < dim; ++row) {
    permutation.indices()(row) = indices(row);
  }
  return permutation;
}

Matrix permuteSymmetricMatrix(
    const Matrix& matrix,
    const Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>&
        permutation) {
  return permutation.transpose() * matrix * permutation;
}

Matrix oneFoot(const Vector3& foothold) {
  Matrix footholds(3, 1);
  footholds.col(0) = foothold;
  return footholds;
}

Matrix twoFootholds() { return Matrix{{1.0, -0.2}, {0.3, 0.4}, {0.5, -0.1}}; }

Matrix fourFootholds() { return Matrix::Zero(3, 4); }

NavState identityState() {
  return NavState(Rot3(), Point3(0.0, 0.0, 0.0), Vector3::Zero());
}

NavState sampleNavState() {
  return NavState(Rot3::RzRyRx(0.1, -0.2, 0.05), Point3(0.2, -0.4, 0.3),
                  Vector3(0.6, -0.1, 0.2));
}

NavState navStateFromEstimate(const ExtendedPose3d& estimate) {
  return NavState(estimate.rotation(), estimate.x(0), estimate.x(1));
}

Matrix footholdsFromEstimate(const ExtendedPose3d& estimate) {
  const Eigen::Index numFeet =
      static_cast<Eigen::Index>(estimate.k() - static_cast<size_t>(2));
  return estimate.xMatrix().rightCols(numFeet);
}

Matrix fullContactFilterCovariance(double poseSigma, double velocitySigma,
                                   double footholdSigma) {
  Matrix covariance = Matrix::Zero(21, 21);
  covariance.block(0, 0, 6, 6) =
      Matrix::Identity(6, 6) * (poseSigma * poseSigma);
  covariance.block(6, 6, 3, 3) = I_3x3 * (velocitySigma * velocitySigma);
  covariance.block(9, 9, 12, 12) =
      Matrix::Identity(12, 12) * (footholdSigma * footholdSigma);
  return covariance;
}

Matrix9 fullContactBaseCovariance(double poseSigma, double velocitySigma) {
  Matrix9 covariance = Matrix9::Zero();
  covariance.topLeftCorner<6, 6>() =
      Matrix::Identity(6, 6) * (poseSigma * poseSigma);
  covariance.bottomRightCorner<3, 3>() =
      I_3x3 * (velocitySigma * velocitySigma);
  return covariance;
}

std::vector<ContactMeasurement> realReplayContacts(size_t eventIndex) {
  switch (eventIndex) {
    case 0:
      return {{0, Vector3(0.374083578587, 0.130614116788, -0.503917515278)},
              {1, Vector3(0.249635994434, -0.170448854566, -0.510778665543)},
              {2, Vector3(-0.347598612309, 0.177350610495, -0.507786512375)},
              {3, Vector3(-0.341215342283, -0.164554029703, -0.511552393436)}};
    case 1:
      return {{0, Vector3(0.140942052007, 0.116847425699, -0.497972786427)},
              {1, Vector3(0.429882109165, -0.062552683055, -0.505733430386)},
              {2, Vector3(-0.162623643875, 0.106095127761, -0.499895572662)},
              {3, Vector3(-0.568916738033, -0.166614994407, -0.492177367210)}};
    case 2:
      return {{0, Vector3(0.455747604370, 0.078563340008, -0.505808234215)},
              {3, Vector3(-0.145139902830, -0.099087588489, -0.511903226376)}};
    case 3:
      return {{1, Vector3(0.481080681086, -0.087105244398, -0.510063111782)},
              {2, Vector3(-0.115411847830, 0.083123117685, -0.506291151047)}};
    default:
      throw std::invalid_argument("Unsupported real replay event index.");
  }
}

LeggedEstimatorParams makeRealReplayParams() {
  LeggedEstimatorParams params = makeParams();
  params.body_P_imu = Pose3(Rot3(), Point3(0.3, 0.0, 0.15));
  params.footholdInitSigma = 0.282842712474619;
  params.contactCovariance = I_3x3 * (0.005 * 0.005);
  params.heightPriorSigma = 0.05;
  return params;
}

std::unique_ptr<LeggedEstimator> makeOneFootEstimator(
    size_t variant, const LeggedEstimatorParams& params) {
  const NavState initialState = identityState();
  const Matrix footholds = oneFoot(Vector3(1.0, 0.0, 0.0));
  switch (variant) {
    case 0:
      return std::make_unique<LeggedInvariantEKF>(initialState, footholds,
                                                  initialCovariance(1), params);
    case 1:
      return std::make_unique<LeggedInvariantIEKF>(
          initialState, footholds, initialCovariance(1), params);
    case 2:
      return std::make_unique<LeggedFixedLagSmoother>(initialState, footholds,
                                                      I_9x9 * 0.1, params, 1.0);
    case 3:
      return std::make_unique<LeggedCombinedFixedLagSmoother>(
          initialState, footholds, I_9x9 * 0.1, params, 1.0);
    default:
      throw std::invalid_argument("Unsupported test estimator variant.");
  }
}

std::unique_ptr<LeggedEstimator> makeFourPointEstimator(
    size_t variant, const LeggedEstimatorParams& params) {
  const NavState initialState = identityState();
  const Matrix footholds = (Matrix(3, 4) << 0.3, 0.3, -0.2, -0.2, 0.15, -0.15,
                            0.15, -0.15, -0.5, -0.5, -0.5, -0.5)
                               .finished();
  const std::vector<std::string> names{"fl", "fr", "bl", "br"};
  switch (variant) {
    case 0:
      return std::make_unique<LeggedInvariantEKF>(
          initialState, footholds, initialCovariance(4), params, names);
    case 1:
      return std::make_unique<LeggedInvariantIEKF>(
          initialState, footholds, initialCovariance(4), params, names);
    case 2:
      return std::make_unique<LeggedFixedLagSmoother>(
          initialState, footholds, I_9x9 * 0.1, params, 1.0, names);
    case 3:
      return std::make_unique<LeggedCombinedFixedLagSmoother>(
          initialState, footholds, I_9x9 * 0.1, params, 1.0, names);
    default:
      throw std::invalid_argument("Unsupported test estimator variant.");
  }
}

CorrelatedFourPointContactMeasurement fourPointContact(
    const CorrelatedFourPointContactCovariance& covariance, bool touchdown,
    double perturbation) {
  CorrelatedFourPointContactMeasurement group;
  const std::array<Vector3, kCorrelatedContactPointCount> points{
      Vector3(0.30, 0.15, -0.50), Vector3(0.30, -0.15, -0.50),
      Vector3(-0.20, 0.15, -0.50), Vector3(-0.20, -0.15, -0.50)};
  for (size_t point = 0; point < points.size(); ++point) {
    Vector3 measurement = points.at(point);
    measurement.x() += perturbation * static_cast<double>(point + 1);
    measurement.y() -= perturbation * static_cast<double>(point % 2);
    group.points.at(point) = ContactMeasurement{point, measurement, touchdown};
  }
  group.positionCovariance = covariance;
  return group;
}

CorrelatedFourPointContactCovariance correlatedFourPointCovariance(
    double correlation) {
  CorrelatedFourPointContactCovariance covariance =
      CorrelatedFourPointContactCovariance::Identity() * 0.0025;
  for (size_t first = 0; first < kCorrelatedContactPointCount; ++first) {
    for (size_t second = first + 1; second < kCorrelatedContactPointCount;
         ++second) {
      for (Eigen::Index axis = 0; axis < 3; ++axis) {
        const Eigen::Index row = 3 * static_cast<Eigen::Index>(first) + axis;
        const Eigen::Index column =
            3 * static_cast<Eigen::Index>(second) + axis;
        covariance(row, column) = 0.0025 * correlation;
        covariance(column, row) = covariance(row, column);
      }
    }
  }
  return covariance;
}

void applyFourPointTestSequence(
    LeggedEstimator* estimator,
    const CorrelatedFourPointContactCovariance& covariance) {
  estimator->processCorrelatedContacts(
      {fourPointContact(covariance, true, 0.0)});
  estimator->predict(Vector3(0.04, -0.02, 0.01), Vector3(0.8, -0.3, 9.81), 0.1);
  estimator->processCorrelatedContacts(
      {fourPointContact(covariance, false, 0.02)});
}

CorrelatedFourPointContactMeasurement fourPointVelocityContact(
    const CorrelatedFourPointVelocityCovariance& covariance, bool touchdown,
    double perturbation) {
  CorrelatedFourPointContactMeasurement group =
      fourPointContact(covariance.topLeftCorner<kCorrelatedContactDimension,
                                                kCorrelatedContactDimension>(),
                       touchdown, perturbation);
  group.footOriginVelocity = CorrelatedFourPointVelocityMeasurement{
      Vector3(0.02, 0.0, -0.48), Vector3(0.03, -0.01, 0.02),
      Vector3(0.04, -0.02, 0.01), covariance};
  return group;
}

void applyFourPointVelocityTestSequence(
    LeggedEstimator* estimator,
    const CorrelatedFourPointVelocityCovariance& covariance) {
  estimator->processCorrelatedContacts(
      {fourPointVelocityContact(covariance, true, 0.0)});
  estimator->predict(Vector3(0.04, -0.02, 0.01), Vector3(0.8, -0.3, 9.81), 0.1);
  estimator->processCorrelatedContacts(
      {fourPointVelocityContact(covariance, false, 0.02)});
}

void applyCovarianceTestSequence(
    LeggedEstimator* estimator,
    const std::optional<Matrix3>& positionCovariance) {
  estimator->processContacts(
      {ContactMeasurement{0, Vector3(1.0, 0.0, 0.0), true}});
  estimator->predict(Vector3(0.04, -0.02, 0.01), Vector3(0.8, -0.3, 9.81), 0.1);
  estimator->processContacts({ContactMeasurement{0, Vector3(0.75, 0.15, -0.05),
                                                 false, positionCovariance}});
}

PointContactVelocityMeasurement pointVelocityMeasurement(
    const Matrix6& covariance) {
  PointContactVelocityMeasurement measurement;
  measurement.bodyPointVelocity = Vector3(0.03, -0.01, 0.02);
  measurement.angularVelocityBody = Vector3(0.04, -0.02, 0.01);
  measurement.positionVelocityCovariance = covariance;
  return measurement;
}

void applyPointVelocityTestSequence(LeggedEstimator* estimator,
                                    const Matrix6& covariance) {
  estimator->processContacts(
      {ContactMeasurement{0, Vector3(1.0, 0.0, 0.0), true}});
  estimator->predict(Vector3(0.04, -0.02, 0.01), Vector3(0.8, -0.3, 9.81), 0.1);
  estimator->processContacts(
      {ContactMeasurement{0, Vector3(0.75, 0.15, -0.05), false, std::nullopt,
                          pointVelocityMeasurement(covariance)}});
}

}  // namespace

/* ************************************************************************* */
TEST(LeggedEstimator, PredictSubtractsConfiguredImuBias) {
  LeggedEstimatorParams biasedParams = makeParams();
  biasedParams.imuBias = imuBias::ConstantBias(Vector3(0.05, -0.04, 0.03),
                                               Vector3(0.02, -0.01, 0.03));
  const LeggedEstimatorParams unbiasedParams = makeParams();

  const NavState X0 = sampleNavState();
  const Matrix footholds = twoFootholds();
  const Matrix P0 = initialCovariance(2);
  LeggedInvariantEKF biasedEstimator(X0, footholds, P0, biasedParams);
  LeggedInvariantEKF unbiasedEstimator(X0, footholds, P0, unbiasedParams);

  const Vector3 omegaBody(0.2, -0.3, 0.1);
  const Vector3 specificForceBody(0.4, -0.2, 0.5);
  const double dt = 0.02;

  biasedEstimator.predict(
      omegaBody + biasedParams.imuBias.gyroscope(),
      specificForceBody + biasedParams.imuBias.accelerometer(), dt);
  unbiasedEstimator.predict(omegaBody, specificForceBody, dt);

  EXPECT(assert_equal(navStateFromEstimate(unbiasedEstimator.estimate()),
                      navStateFromEstimate(biasedEstimator.estimate()), 1e-9));
  EXPECT(assert_equal(footholdsFromEstimate(unbiasedEstimator.estimate()),
                      footholdsFromEstimate(biasedEstimator.estimate()),
                      1e-12));
}

/* ************************************************************************* */
TEST(LeggedEstimator, PerContactCovarianceFallbackMatchesConfiguredValue) {
  const LeggedEstimatorParams params = makeParams();
  for (size_t variant = 0; variant < 4; ++variant) {
    auto fallback = makeOneFootEstimator(variant, params);
    auto explicitValue = makeOneFootEstimator(variant, params);

    applyCovarianceTestSequence(fallback.get(), std::nullopt);
    applyCovarianceTestSequence(explicitValue.get(), params.contactCovariance);

    EXPECT(
        assert_equal(fallback->estimate(), explicitValue->estimate(), 1e-10));
  }
}

/* ************************************************************************* */
TEST(LeggedEstimator, PerContactCovarianceReachesAllFourCores) {
  const LeggedEstimatorParams params = makeParams();
  const Matrix3 tightCovariance = I_3x3 * 1e-8;
  const Matrix3 looseCovariance = I_3x3 * 1e2;
  for (size_t variant = 0; variant < 4; ++variant) {
    auto tight = makeOneFootEstimator(variant, params);
    auto loose = makeOneFootEstimator(variant, params);

    applyCovarianceTestSequence(tight.get(), tightCovariance);
    applyCovarianceTestSequence(loose.get(), looseCovariance);

    const Vector6 difference =
        tight->estimate().localCoordinates(loose->estimate()).head<6>();
    CHECK(difference.norm() > 1e-7);
  }
}

/* ************************************************************************* */
TEST(LeggedEstimator, CorrelatedFourPointCovarianceReachesBothFilters) {
  const LeggedEstimatorParams params = makeParams();
  const CorrelatedFourPointContactCovariance independent =
      correlatedFourPointCovariance(0.0);
  const CorrelatedFourPointContactCovariance correlated =
      correlatedFourPointCovariance(0.35);
  for (size_t variant = 0; variant < 2; ++variant) {
    auto independentEstimator = makeFourPointEstimator(variant, params);
    auto correlatedEstimator = makeFourPointEstimator(variant, params);

    applyFourPointTestSequence(independentEstimator.get(), independent);
    applyFourPointTestSequence(correlatedEstimator.get(), correlated);

    const Vector9 difference =
        navStateFromEstimate(independentEstimator->estimate())
            .localCoordinates(
                navStateFromEstimate(correlatedEstimator->estimate()));
    CHECK(difference.norm() > 1e-9);
  }
}

/* ************************************************************************* */
TEST(LeggedEstimator, GroupedContactsLinearizeAfterAllHeightCorrections) {
  const LeggedEstimatorParams params = makeParams();
  const NavState navState = sampleNavState();
  const Matrix footholds = Matrix::Zero(3, 4);
  const Matrix priorCovariance = denseCorrelatedCovariance(4) * 1e-3;
  const double terrainHeight = 0.2;

  for (const bool includesVelocity : {false, true}) {
    ExposedLeggedInvariantEKF actual(navState, footholds, priorCovariance,
                                     params);
    actual.turnHeightPriorOn(terrainHeight);
    LeftLinearEKF<ExtendedPose3d> expected(actual.estimate(), priorCovariance);
    const auto group =
        includesVelocity
            ? fourPointVelocityContact(
                  CorrelatedFourPointVelocityCovariance::Identity() * 0.01,
                  false, 0.02)
            : fourPointContact(correlatedFourPointCovariance(0.35), false,
                               0.02);

    // Complete the sequential height corrections before forming one joint
    // residual.
    for (size_t point = 0; point < kCorrelatedContactPointCount; ++point) {
      ExtendedPoseHeightFactor height(0, 2 + point, terrainHeight,
                                      noiseModel::Unit::Create(1));
      Matrix H;
      const Vector residual = height.evaluateError(expected.state(), H);
      expected.updateWithVector(
          residual, H, Vector1::Zero(),
          I_1x1 * params.heightPriorSigma * params.heightPriorSigma);
    }

    std::array<Point3, kCorrelatedContactPointCount> measurements;
    for (size_t point = 0; point < measurements.size(); ++point) {
      measurements[point] = group.points[point].bodyPoint;
    }
    Matrix H;
    Vector residual;
    Matrix covariance;
    if (includesVelocity) {
      const auto& velocity = *group.footOriginVelocity;
      ExtendedPoseFourPointVelocityContactFactor contact(
          0, {2, 3, 4, 5}, measurements, velocity.bodyPoint,
          velocity.bodyPointVelocity, velocity.angularVelocityBody,
          noiseModel::Unit::Create(15));
      residual = contact.evaluateError(expected.state(), H);
      covariance = velocity.positionVelocityCovariance;
    } else {
      ExtendedPoseFourPointContactFactor contact(0, {2, 3, 4, 5}, measurements,
                                                 noiseModel::Unit::Create(12));
      residual = contact.evaluateError(expected.state(), H);
      covariance = group.positionCovariance;
    }
    expected.updateWithVector(residual, H, Vector::Zero(residual.size()),
                              covariance);

    actual.applyCorrelatedContactUpdate({group});

    EXPECT(assert_equal(expected.state(), actual.estimate(), 1e-10));
    EXPECT(assert_equal(expected.covariance(), actual.covariance(), 1e-10));
  }
}

/* ************************************************************************* */
TEST(LeggedEstimator,
     CorrelatedFourPointVelocityMeasurementReachesBothFilters) {
  const LeggedEstimatorParams params = makeParams();
  CorrelatedFourPointVelocityCovariance tight =
      CorrelatedFourPointVelocityCovariance::Identity() * 1e-8;
  tight.topLeftCorner<kCorrelatedContactDimension,
                      kCorrelatedContactDimension>() =
      correlatedFourPointCovariance(0.2);
  CorrelatedFourPointVelocityCovariance loose = tight;
  loose.bottomRightCorner<3, 3>() = I_3x3 * 1e2;
  for (size_t variant = 0; variant < 2; ++variant) {
    auto tightEstimator = makeFourPointEstimator(variant, params);
    auto looseEstimator = makeFourPointEstimator(variant, params);

    applyFourPointVelocityTestSequence(tightEstimator.get(), tight);
    applyFourPointVelocityTestSequence(looseEstimator.get(), loose);

    const Vector9 difference =
        navStateFromEstimate(tightEstimator->estimate())
            .localCoordinates(navStateFromEstimate(looseEstimator->estimate()));
    CHECK(difference.norm() > 1e-7);
  }
}

/* ************************************************************************* */
TEST(LeggedEstimator, RejectsMalformedCorrelatedFourPointPackets) {
  CorrelatedFourPointContactCovariance nonFinite =
      CorrelatedFourPointContactCovariance::Identity();
  nonFinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
  CorrelatedFourPointContactCovariance nonSymmetric =
      CorrelatedFourPointContactCovariance::Identity();
  nonSymmetric(0, 1) = 0.1;
  CorrelatedFourPointContactCovariance indefinite =
      CorrelatedFourPointContactCovariance::Identity();
  indefinite(0, 0) = -1.0;

  const LeggedEstimatorParams params = makeParams();
  for (size_t variant = 0; variant < 2; ++variant) {
    for (const CorrelatedFourPointContactCovariance& covariance :
         {nonFinite, nonSymmetric, indefinite}) {
      auto estimator = makeFourPointEstimator(variant, params);
      CHECK_EXCEPTION(estimator->processCorrelatedContacts(
                          {fourPointContact(covariance, false, 0.0)}),
                      std::invalid_argument);

      auto velocityGroup = fourPointVelocityContact(
          CorrelatedFourPointVelocityCovariance::Identity(), false, 0.0);
      velocityGroup.positionCovariance = covariance;
      CHECK_EXCEPTION(estimator->processCorrelatedContacts({velocityGroup}),
                      std::invalid_argument);
      velocityGroup.footOriginVelocity->positionVelocityCovariance
          .topLeftCorner<12, 12>() = covariance;
      CHECK_EXCEPTION(estimator->processCorrelatedContacts({velocityGroup}),
                      std::invalid_argument);
    }

    auto estimator = makeFourPointEstimator(variant, params);
    CorrelatedFourPointContactMeasurement mixedTouchdown = fourPointContact(
        CorrelatedFourPointContactCovariance::Identity(), false, 0.0);
    mixedTouchdown.points.at(2).touchdown = true;
    CHECK_EXCEPTION(estimator->processCorrelatedContacts({mixedTouchdown}),
                    std::invalid_argument);

    CorrelatedFourPointContactMeasurement independentOverride =
        fourPointContact(CorrelatedFourPointContactCovariance::Identity(),
                         false, 0.0);
    independentOverride.points.at(0).positionCovariance = I_3x3;
    CHECK_EXCEPTION(estimator->processCorrelatedContacts({independentOverride}),
                    std::invalid_argument);

    CorrelatedFourPointVelocityCovariance mismatched =
        CorrelatedFourPointVelocityCovariance::Identity();
    CorrelatedFourPointContactMeasurement mismatchedVelocity =
        fourPointVelocityContact(mismatched, false, 0.0);
    mismatchedVelocity.positionCovariance(0, 0) = 2.0;
    CHECK_EXCEPTION(estimator->processCorrelatedContacts({mismatchedVelocity}),
                    std::invalid_argument);

    // Approximate block equality alone does not guarantee positive
    // definiteness.
    auto nearSingular = fourPointVelocityContact(
        CorrelatedFourPointVelocityCovariance::Identity(), false, 0.0);
    nearSingular.footOriginVelocity->positionVelocityCovariance(0, 0) = 1e-14;
    nearSingular.positionCovariance(0, 0) = -1e-14;
    CHECK_EXCEPTION(estimator->processCorrelatedContacts({nearSingular}),
                    std::invalid_argument);
  }
}

/* ************************************************************************* */
TEST(LeggedEstimator, RejectsInvalidPerContactCovarianceInAllFourCores) {
  Matrix3 nonFinite = I_3x3;
  nonFinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
  Matrix3 nonSymmetric = I_3x3;
  nonSymmetric(0, 1) = 0.1;
  const Matrix3 indefinite =
      (Matrix3() << 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished();

  const LeggedEstimatorParams params = makeParams();
  for (size_t variant = 0; variant < 4; ++variant) {
    for (const Matrix3& covariance : {nonFinite, nonSymmetric, indefinite}) {
      auto estimator = makeOneFootEstimator(variant, params);
      const ContactMeasurement contact(0, Vector3(1.0, 0.0, 0.0), false,
                                       covariance);
      CHECK_EXCEPTION(estimator->processContacts({contact}),
                      std::invalid_argument);
    }
  }
}

/* ************************************************************************* */
TEST(LeggedEstimator, RejectsInvalidConfiguredContactCovariance) {
  Matrix3 nonFinite = I_3x3;
  nonFinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
  Matrix3 nonSymmetric = I_3x3;
  nonSymmetric(0, 1) = 0.1;
  const Matrix3 indefinite =
      (Matrix3() << 1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0).finished();

  for (size_t variant = 0; variant < 4; ++variant) {
    for (const Matrix3& covariance : {nonFinite, nonSymmetric, indefinite}) {
      LeggedEstimatorParams params = makeParams();
      params.contactCovariance = covariance;
      CHECK_EXCEPTION(makeOneFootEstimator(variant, params),
                      std::invalid_argument);
    }
  }
}

/* ************************************************************************* */
TEST(LeggedEstimator,
     JointContactsPreserveFramesAndPositionVelocityCorrelation) {
  LeggedEstimatorParams params = makeParams();
  params.imuBias =
      imuBias::ConstantBias(Vector3::Zero(), Vector3(0.01, -0.02, 0.03));
  LeggedEstimatorParams transformedParams = params;
  const Pose3 body_P_imu(Rot3::RzRyRx(0.2, -0.3, 0.4), Point3(0.3, -0.1, 0.15));
  transformedParams.body_P_imu = body_P_imu;

  for (size_t variant = 0; variant < 4; ++variant) {
    for (const bool grouped : {false, true}) {
      if (grouped && variant >= 2) continue;
      auto reference = makeFourPointEstimator(variant, params);
      auto transformed = makeFourPointEstimator(variant, transformedParams);
      auto decorrelated = makeFourPointEstimator(variant, params);

      if (grouped) {
        CorrelatedFourPointVelocityCovariance covariance =
            CorrelatedFourPointVelocityCovariance::Identity() * 0.01;
        covariance(0, 12) = covariance(12, 0) = 0.003;
        covariance(1, 13) = covariance(13, 1) = -0.002;
        const auto group = fourPointVelocityContact(covariance, true, 0.02);
        auto bodyGroup = group;
        for (auto& point : bodyGroup.points) {
          point.bodyPoint = body_P_imu.transformFrom(point.bodyPoint);
        }
        auto& velocity = *bodyGroup.footOriginVelocity;
        velocity.bodyPoint = body_P_imu.transformFrom(velocity.bodyPoint);
        velocity.bodyPointVelocity =
            body_P_imu.rotation().rotate(velocity.bodyPointVelocity);
        velocity.angularVelocityBody =
            body_P_imu.rotation().rotate(velocity.angularVelocityBody);

        reference->processCorrelatedContacts({group});
        transformed->processCorrelatedContacts({bodyGroup});
        covariance.topRightCorner<12, 3>().setZero();
        covariance.bottomLeftCorner<3, 12>().setZero();
        decorrelated->processCorrelatedContacts(
            {fourPointVelocityContact(covariance, true, 0.02)});
      } else {
        Matrix6 covariance = Matrix6::Identity() * 0.01;
        covariance(0, 3) = covariance(3, 0) = 0.003;
        covariance(1, 4) = covariance(4, 1) = -0.002;
        const ContactMeasurement contact(0, Vector3(0.32, 0.15, -0.5), true,
                                         std::nullopt,
                                         pointVelocityMeasurement(covariance));
        auto bodyContact = contact;
        bodyContact.bodyPoint = body_P_imu.transformFrom(bodyContact.bodyPoint);
        auto& velocity = *bodyContact.pointVelocity;
        velocity.bodyPointVelocity =
            body_P_imu.rotation().rotate(velocity.bodyPointVelocity);
        velocity.angularVelocityBody =
            body_P_imu.rotation().rotate(velocity.angularVelocityBody);

        reference->processContacts({contact});
        transformed->processContacts({bodyContact});
        auto independentContact = contact;
        auto& independentCovariance =
            independentContact.pointVelocity->positionVelocityCovariance;
        independentCovariance.topRightCorner<3, 3>().setZero();
        independentCovariance.bottomLeftCorner<3, 3>().setZero();
        decorrelated->processContacts({independentContact});
      }

      // Re-expressing body measurements must not rotate the IMU-frame
      // covariance.
      EXPECT(
          assert_equal(reference->estimate(), transformed->estimate(), 1e-8));
      CHECK(reference->estimate()
                .localCoordinates(decorrelated->estimate())
                .norm() > 1e-7);
    }
  }
}

/* ************************************************************************* */
TEST(LeggedEstimator, PointVelocityMeasurementReachesAllFourCores) {
  const LeggedEstimatorParams params = makeParams();
  Matrix6 tight = Matrix6::Identity() * 1e-8;
  tight.topLeftCorner<3, 3>() = params.contactCovariance;
  Matrix6 loose = tight;
  loose.bottomRightCorner<3, 3>() = I_3x3 * 1e2;
  for (size_t variant = 0; variant < 4; ++variant) {
    auto tightEstimator = makeOneFootEstimator(variant, params);
    auto looseEstimator = makeOneFootEstimator(variant, params);

    applyPointVelocityTestSequence(tightEstimator.get(), tight);
    applyPointVelocityTestSequence(looseEstimator.get(), loose);

    const Vector9 difference =
        navStateFromEstimate(tightEstimator->estimate())
            .localCoordinates(navStateFromEstimate(looseEstimator->estimate()));
    CHECK(difference.norm() > 1e-7);
  }
}

/* ************************************************************************* */
TEST(LeggedEstimator, RejectsInvalidPointVelocityMeasurementInAllFourCores) {
  Matrix6 nonFinite = Matrix6::Identity();
  nonFinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
  Matrix6 nonSymmetric = Matrix6::Identity();
  nonSymmetric(0, 3) = 0.1;
  Matrix6 indefinite = Matrix6::Identity();
  indefinite(0, 0) = -1.0;

  const LeggedEstimatorParams params = makeParams();
  for (size_t variant = 0; variant < 4; ++variant) {
    for (const Matrix6& covariance : {nonFinite, nonSymmetric, indefinite}) {
      auto estimator = makeOneFootEstimator(variant, params);
      const ContactMeasurement contact(0, Vector3(1.0, 0.0, 0.0), false,
                                       std::nullopt,
                                       pointVelocityMeasurement(covariance));
      CHECK_EXCEPTION(estimator->processContacts({contact}),
                      std::invalid_argument);
    }
    auto estimator = makeOneFootEstimator(variant, params);
    CHECK_EXCEPTION(estimator->processContacts({ContactMeasurement{
                        0, Vector3(1.0, 0.0, 0.0), false, I_3x3,
                        pointVelocityMeasurement(Matrix6::Identity())}}),
                    std::invalid_argument);
  }
}

/* ************************************************************************* */
TEST(LeggedEstimator, FullContactInitializationKeepsUnobservedFilterVelocity) {
  LeggedEstimatorParams params = makeRealReplayParams();
  params.useFullContactInitialization = true;
  const NavState priorState(Rot3(), Point3(0.0, 0.0, 0.72),
                            Vector3(0.35, -0.10, 0.08));
  const Matrix P0 = fullContactFilterCovariance(0.2, 1e-3, 0.5);
  LeggedInvariantEKF estimator(priorState, fourFootholds(), P0, params,
                               {"FL", "FR", "RL", "RR"});

  estimator.processContacts(realReplayContacts(0));

  const NavState initializedState = navStateFromEstimate(estimator.estimate());
  EXPECT(
      assert_equal(priorState.velocity(), initializedState.velocity(), 1e-6));
}

/* ************************************************************************* */
TEST(LeggedEstimator, FilterWaitsForFullContactInitialization) {
  LeggedEstimatorParams params = makeRealReplayParams();
  params.useFullContactInitialization = true;
  const NavState priorState = sampleNavState();
  LeggedInvariantEKF estimator(priorState, fourFootholds(),
                               fullContactFilterCovariance(0.2, 0.1, 0.5),
                               params, {"FL", "FR", "RL", "RR"});

  estimator.predict(Vector3(0.3, -0.1, 0.2), Vector3(0.2, -0.4, 9.7), 0.2);
  estimator.processContacts(realReplayContacts(2));

  EXPECT(assert_equal(priorState, navStateFromEstimate(estimator.estimate()),
                      1e-12));
  EXPECT(assert_equal(fourFootholds(),
                      footholdsFromEstimate(estimator.estimate()), 1e-12));
}

/* ************************************************************************* */
TEST(LeggedEstimator, FixedLagSmootherWaitsForFullContactInitialization) {
  LeggedEstimatorParams params = makeRealReplayParams();
  params.useFullContactInitialization = true;
  const NavState priorState = sampleNavState();
  LeggedFixedLagSmoother estimator(priorState, fourFootholds(),
                                   fullContactBaseCovariance(0.2, 0.1), params,
                                   0.15, {"FL", "FR", "RL", "RR"});

  estimator.predict(Vector3(0.3, -0.1, 0.2), Vector3(0.2, -0.4, 9.7), 0.2);
  estimator.processContacts(realReplayContacts(2));

  EXPECT(assert_equal(priorState, navStateFromEstimate(estimator.estimate()),
                      1e-12));
  EXPECT(assert_equal(fourFootholds(),
                      footholdsFromEstimate(estimator.estimate()), 1e-12));
}

/* ************************************************************************* */
TEST(LeggedEstimator, FullContactInitializationFilterDependsOnPriorCovariance) {
  LeggedEstimatorParams params = makeRealReplayParams();
  params.useFullContactInitialization = true;
  const double priorPitch = 0.25;
  const NavState priorState(Rot3::Ypr(0.0, priorPitch, 0.0),
                            Point3(0.0, 0.0, 0.72), Vector3::Zero());

  LeggedInvariantEKF tightEstimator(
      priorState, fourFootholds(), fullContactFilterCovariance(0.02, 0.05, 0.5),
      params, {"FL", "FR", "RL", "RR"});
  LeggedInvariantEKF looseEstimator(priorState, fourFootholds(),
                                    fullContactFilterCovariance(2.0, 0.05, 0.5),
                                    params, {"FL", "FR", "RL", "RR"});

  tightEstimator.processContacts(realReplayContacts(0));
  looseEstimator.processContacts(realReplayContacts(0));

  const double tightPitch =
      navStateFromEstimate(tightEstimator.estimate()).attitude().rpy().y();
  const double loosePitch =
      navStateFromEstimate(looseEstimator.estimate()).attitude().rpy().y();
  CHECK(std::abs(tightPitch - loosePitch) > 1e-2);
  CHECK(std::abs(tightPitch - priorPitch) < std::abs(loosePitch - priorPitch));
}

/* ************************************************************************* */
TEST(LeggedEstimator,
     FullContactInitializationFixedLagSmootherDependsOnPriorCovariance) {
  LeggedEstimatorParams params = makeRealReplayParams();
  params.useFullContactInitialization = true;
  const double priorPitch = 0.25;
  const NavState priorState(Rot3::Ypr(0.0, priorPitch, 0.0),
                            Point3(0.0, 0.0, 0.72), Vector3::Zero());

  LeggedFixedLagSmoother tightEstimator(priorState, fourFootholds(),
                                        fullContactBaseCovariance(0.02, 0.05),
                                        params, 0.15, {"FL", "FR", "RL", "RR"});
  LeggedFixedLagSmoother looseEstimator(priorState, fourFootholds(),
                                        fullContactBaseCovariance(2.0, 0.05),
                                        params, 0.15, {"FL", "FR", "RL", "RR"});

  tightEstimator.processContacts(realReplayContacts(0));
  looseEstimator.processContacts(realReplayContacts(0));

  const double tightPitch =
      navStateFromEstimate(tightEstimator.estimate()).attitude().rpy().y();
  const double loosePitch =
      navStateFromEstimate(looseEstimator.estimate()).attitude().rpy().y();
  CHECK(std::abs(tightPitch - loosePitch) > 1e-2);
  CHECK(std::abs(tightPitch - priorPitch) < std::abs(loosePitch - priorPitch));
}

/* ************************************************************************* */
TEST(LeggedEstimator,
     FullContactInitializationCombinedSmootherDependsOnPriorCovariance) {
  LeggedEstimatorParams params = makeRealReplayParams();
  params.useFullContactInitialization = true;
  const double priorPitch = 0.25;
  const NavState priorState(Rot3::Ypr(0.0, priorPitch, 0.0),
                            Point3(0.0, 0.0, 0.72), Vector3::Zero());

  LeggedCombinedFixedLagSmoother tightEstimator(
      priorState, fourFootholds(), fullContactBaseCovariance(0.02, 0.05),
      params, 0.15, {"FL", "FR", "RL", "RR"});
  LeggedCombinedFixedLagSmoother looseEstimator(
      priorState, fourFootholds(), fullContactBaseCovariance(2.0, 0.05), params,
      0.15, {"FL", "FR", "RL", "RR"});

  tightEstimator.processContacts(realReplayContacts(0));
  looseEstimator.processContacts(realReplayContacts(0));

  const double tightPitch =
      navStateFromEstimate(tightEstimator.estimate()).attitude().rpy().y();
  const double loosePitch =
      navStateFromEstimate(looseEstimator.estimate()).attitude().rpy().y();
  CHECK(std::abs(tightPitch - loosePitch) > 1e-2);
  CHECK(std::abs(tightPitch - priorPitch) < std::abs(loosePitch - priorPitch));
}

/* ************************************************************************* */
TEST(LeggedEstimator, LeggedInvariantDynamicsJacobian) {
  const NavState X0 = sampleNavState();
  const Matrix footholds0 = twoFootholds();
  const ExtendedPose3d state = LeggedInvariantEKF::MakeState(X0, footholds0);
  const Vector3 omegaBody(0.2, -0.3, 0.1);
  const Vector3 specificForceBody(0.4, -0.2, 0.5);
  const Vector3 gravity(0.0, 0.0, -9.81);
  const double dt = 0.02;

  const ExtendedPose3d W =
      LeggedInvariantEKF::GravityIncrement(kTwoFeet, gravity, dt);
  const ExtendedPose3d U = LeggedInvariantEKF::ImuIncrement(
      kTwoFeet, omegaBody, specificForceBody, dt);
  const LeggedInvariantEKF::AutonomousFlow phi(kTwoFeet, dt);

  Matrix A;
  (void)LeftLinearEKF<ExtendedPose3d>::template Dynamics<
      LeggedInvariantEKF::AutonomousFlow>(W, phi, state, U, A);
  const auto f = [&](const ExtendedPose3d& X) {
    return LeftLinearEKF<ExtendedPose3d>::template Dynamics<
        LeggedInvariantEKF::AutonomousFlow>(W, phi, X, U);
  };
  const Matrix expected =
      numericalDerivative11<ExtendedPose3d, ExtendedPose3d, kExtendedPoseDim>(
          f, state, 1e-6);

  EXPECT(assert_equal(expected, A, 1e-6));
}

/* ************************************************************************* */
TEST(LeggedEstimator, MarginalizeLeavingFootResetsCovarianceBlock) {
  const LeggedEstimatorParams params = makeParams(true);
  LeggedInvariantEKF estimator(identityState(), oneFoot(Vector3(1.0, 0.0, 0.0)),
                               initialCovariance(1), params);

  estimator.processContacts({ContactMeasurement{0, Vector3(1.0, 0.0, 0.0)}});
  const Matrix afterContact = estimator.covariance();
  CHECK(afterContact.block(0, 9, 9, 3).norm() > 1e-6);

  estimator.processContacts({});
  const Matrix afterLeaving = estimator.covariance();
  const Matrix3 expectedFoot =
      I_3x3 * (params.footholdInitSigma * params.footholdInitSigma);

  EXPECT_DOUBLES_EQUAL(0.0, afterLeaving.block(0, 9, 9, 3).norm(), 1e-12);
  EXPECT_DOUBLES_EQUAL(0.0, afterLeaving.block(9, 0, 3, 9).norm(), 1e-12);
  EXPECT(assert_equal(Matrix(expectedFoot), afterLeaving.block(9, 9, 3, 3),
                      1e-12));
  EXPECT_DOUBLES_EQUAL(
      0.0, Point3(footholdsFromEstimate(estimator.estimate()).col(0)).norm(),
      1e-12);
}

/* ************************************************************************* */
TEST(LeggedEstimator, MarginalizeFootPreservesRetainedSchurComplement) {
  const size_t numFeet = 2;
  const size_t marginalizedFoot = 0;
  LeggedEstimatorParams params = makeParams(true);
  params.footholdInitSigma = 0.3;
  const Matrix P0 = denseCorrelatedCovariance(numFeet);
  const Matrix footholds{{0.4, -0.2}, {1.1, 0.7}, {-0.5, 0.3}};
  ExposedLeggedInvariantEKF estimator(identityState(), footholds, P0, params);

  estimator.marginalizeFoot(marginalizedFoot);
  const Matrix marginalized = estimator.covariance();

  const int retainedDim = static_cast<int>(P0.rows()) - 3;
  const auto permutation =
      makeMarginalizationPermutation(numFeet, marginalizedFoot);
  const Matrix permutedInformation =
      permuteSymmetricMatrix(P0.inverse(), permutation);
  const Matrix Lambda_rr =
      permutedInformation.topLeftCorner(retainedDim, retainedDim);
  const Matrix Lambda_rl = permutedInformation.topRightCorner(retainedDim, 3);
  const Matrix Lambda_lr = permutedInformation.bottomLeftCorner(3, retainedDim);
  const Matrix Lambda_ll = permutedInformation.bottomRightCorner(3, 3);
  const Matrix expectedRetained =
      (Lambda_rr - Lambda_rl * Lambda_ll.inverse() * Lambda_lr).inverse();
  const Matrix permutedMarginalized =
      permuteSymmetricMatrix(marginalized, permutation);
  const Matrix actualRetained =
      permutedMarginalized.topLeftCorner(retainedDim, retainedDim);
  const Matrix actualCross =
      permutedMarginalized.topRightCorner(retainedDim, 3);
  Matrix expectedPermuted = Matrix::Zero(P0.rows(), P0.cols());
  expectedPermuted.topLeftCorner(retainedDim, retainedDim) = expectedRetained;
  expectedPermuted.bottomRightCorner(3, 3) =
      I_3x3 * (params.footholdInitSigma * params.footholdInitSigma);
  const Matrix expectedMarginalized =
      permutation * expectedPermuted * permutation.transpose();

  EXPECT(assert_equal(expectedRetained, actualRetained, 1e-8));
  EXPECT_DOUBLES_EQUAL(0.0, actualCross.norm(), 1e-12);
  EXPECT_DOUBLES_EQUAL(0.0, actualCross.transpose().norm(), 1e-12);
  EXPECT(assert_equal(expectedMarginalized, marginalized, 1e-8));
}

/* ************************************************************************* */
TEST(LeggedEstimator, TouchdownFlagReinitializesFootWithoutIntermediateSwing) {
  const LeggedEstimatorParams params = makeParams();
  LeggedInvariantEKF estimator(identityState(), oneFoot(Vector3::Zero()),
                               initialCovariance(1), params);

  estimator.processContacts(
      {ContactMeasurement{0, Vector3(1.0, 0.0, 0.0), true}});
  const Point3 firstFoothold =
      Point3(footholdsFromEstimate(estimator.estimate()).col(0));

  estimator.processContacts(
      {ContactMeasurement{0, Vector3(2.0, 0.0, 0.0), true}});
  const Point3 secondFoothold =
      Point3(footholdsFromEstimate(estimator.estimate()).col(0));

  EXPECT(assert_equal(Point3(1.0, 0.0, 0.0), firstFoothold, 1e-6));
  EXPECT(assert_equal(Point3(2.0, 0.0, 0.0), secondFoothold, 1e-6));
}

/* ************************************************************************* */
TEST(LeggedEstimator, HeightPriorChangesInvariantFilterContactUpdate) {
  LeggedEstimatorParams params = makeParams();
  params.body_P_imu = Pose3(Rot3(), Point3(0.30, 0.0, 0.15));
  params.footholdInitSigma = 0.5;
  params.contactCovariance =
      (Vector3(0.03 * 0.03, 0.03 * 0.03, 0.02 * 0.02)).asDiagonal();
  params.heightPriorSigma = 0.02;

  const NavState navState0(Rot3(), Point3(0.0, 0.0, 0.12),
                           Vector3(0.0, 0.0, -0.12));
  Matrix P0 = Matrix::Zero(12, 12);
  P0.block(0, 0, 9, 9) = I_9x9 * 1e-3;
  P0.block(9, 9, 3, 3) =
      I_3x3 * (params.footholdInitSigma * params.footholdInitSigma);

  const ContactMeasurement contact{0, Vector3::Zero(), false};

  LeggedInvariantEKF noPrior(navState0, oneFoot(Vector3::Zero()), P0, params,
                             {"foot"});
  noPrior.turnHeightPriorOff();
  noPrior.processContacts({contact});
  const double noPriorFootHeight =
      Point3(footholdsFromEstimate(noPrior.estimate()).col(0)).z();

  LeggedInvariantEKF withPrior(navState0, oneFoot(Vector3::Zero()), P0, params,
                               {"foot"});
  withPrior.turnHeightPriorOn(10.0);
  withPrior.processContacts({contact});
  const double withPriorFootHeight =
      Point3(footholdsFromEstimate(withPrior.estimate()).col(0)).z();

  CHECK(std::abs(withPriorFootHeight - noPriorFootHeight) > 1.0);
}

/* ************************************************************************* */
TEST(LeggedEstimator, ExtendedPoseSequentialAndGraphUpdatesAgree) {
  const LeggedEstimatorParams params = makeParams();
  const Matrix P0 = initialCovariance(1);
  LeggedInvariantEKF sequential(identityState(),
                                oneFoot(Vector3(1.0, 0.0, 0.0)), P0, params);
  LeggedInvariantIEKF graph(identityState(), oneFoot(Vector3(1.0, 0.0, 0.0)),
                            P0, params);

  const ContactMeasurement first{0, Vector3(1.0, 0.0, 0.0)};
  const ContactMeasurement second{0, Vector3(0.9, 0.1, 0.0)};
  sequential.processContacts({first});
  graph.processContacts({first});
  sequential.processContacts({second});
  graph.processContacts({second});

  EXPECT(assert_equal(navStateFromEstimate(sequential.estimate()),
                      navStateFromEstimate(graph.estimate()), 3e-3));
  EXPECT(assert_equal(footholdsFromEstimate(sequential.estimate()),
                      footholdsFromEstimate(graph.estimate()), 3e-3));
  EXPECT(assert_equal(sequential.covariance(), graph.covariance(), 5e-4));
}

/* ************************************************************************* */
TEST(LeggedEstimator, FixedLagEstimatorReplacesContactEpisode) {
  const LeggedEstimatorParams params = makeParams();
  LeggedFixedLagSmoother estimator(identityState(), oneFoot(Vector3::Zero()),
                                   I_9x9 * 1e-3, params, 0.15);

  estimator.processContacts({ContactMeasurement{0, Vector3(1.0, 0.0, 0.0)}});
  EXPECT(assert_equal(
      Point3(1.0, 0.0, 0.0),
      Point3(footholdsFromEstimate(estimator.estimate()).col(0)), 5e-3));

  const Vector3 stationarySpecificForce(0.0, 0.0, 9.81);
  estimator.predict(Vector3::Zero(), stationarySpecificForce, 0.1);
  const NavState expectedDeadReckoned = NavStateImuEKF::Dynamics(
      params.preintegrationParams->n_gravity, identityState(), Vector3::Zero(),
      stationarySpecificForce, 0.1);
  EXPECT(assert_equal(expectedDeadReckoned,
                      navStateFromEstimate(estimator.estimate()), 1e-6));
  estimator.processContacts({ContactMeasurement{0, Vector3(1.0, 0.0, 0.0)}});

  estimator.predict(Vector3::Zero(), stationarySpecificForce, 0.1);
  estimator.processContacts({});
  EXPECT_DOUBLES_EQUAL(
      0.0, Point3(footholdsFromEstimate(estimator.estimate()).col(0)).norm(),
      1e-12);

  estimator.predict(Vector3::Zero(), stationarySpecificForce, 0.1);
  estimator.processContacts({ContactMeasurement{0, Vector3(2.0, 0.0, 0.0)}});

  EXPECT(assert_equal(
      Point3(2.0, 0.0, 0.0),
      Point3(footholdsFromEstimate(estimator.estimate()).col(0)), 5e-3));
}

/* ************************************************************************* */
TEST(LeggedEstimator, CombinedFixedLagEstimatorReplacesContactEpisode) {
  LeggedEstimatorParams params = makeParams();
  params.imuBias = imuBias::ConstantBias(Vector3(0.02, -0.01, 0.03),
                                         Vector3(0.001, -0.002, 0.0015));
  LeggedCombinedFixedLagSmoother estimator(
      identityState(), oneFoot(Vector3::Zero()), I_9x9 * 1e-3, params, 0.15);

  estimator.processContacts({ContactMeasurement{0, Vector3(1.0, 0.0, 0.0)}});
  EXPECT(assert_equal(
      Point3(1.0, 0.0, 0.0),
      Point3(footholdsFromEstimate(estimator.estimate()).col(0)), 5e-3));

  const Vector3 stationarySpecificForce(0.0, 0.0, 9.81);
  estimator.predict(Vector3::Zero(), stationarySpecificForce, 0.1);
  CHECK(navStateFromEstimate(estimator.estimate()).position().allFinite());
  estimator.processContacts({ContactMeasurement{0, Vector3(1.0, 0.0, 0.0)}});

  estimator.predict(Vector3::Zero(), stationarySpecificForce, 0.1);
  estimator.processContacts({});
  EXPECT_DOUBLES_EQUAL(
      0.0, Point3(footholdsFromEstimate(estimator.estimate()).col(0)).norm(),
      1e-12);

  estimator.predict(Vector3::Zero(), stationarySpecificForce, 0.1);
  estimator.processContacts({ContactMeasurement{0, Vector3(2.0, 0.0, 0.0)}});

  EXPECT(assert_equal(
      Point3(2.0, 0.0, 0.0),
      Point3(footholdsFromEstimate(estimator.estimate()).col(0)), 5e-3));
}

/* ************************************************************************* */
TEST(LeggedEstimator, FixedLagBatchAndIncrementalEndpointsAgree) {
  const LeggedEstimatorParams params = makeParams();
  LeggedFixedLagSmoother batch(identityState(), oneFoot(Vector3(1.0, 0.0, 0.0)),
                               I_9x9 * 1e-3, params, 1.0, {},
                               LeggedFixedLagEngine::Batch);
  LeggedFixedLagSmoother incremental(
      identityState(), oneFoot(Vector3(1.0, 0.0, 0.0)), I_9x9 * 1e-3, params,
      1.0, {}, LeggedFixedLagEngine::Incremental);

  const std::array<ContactMeasurement, 3> contacts{
      ContactMeasurement{0, Vector3(1.0, 0.0, 0.0), true},
      ContactMeasurement{0, Vector3(0.99, 0.01, -0.01)},
      ContactMeasurement{0, Vector3(0.98, 0.02, -0.01)}};
  for (const ContactMeasurement& contact : contacts) {
    batch.predict(Vector3(0.01, -0.02, 0.005), Vector3(0.05, -0.03, 9.81), 0.1);
    incremental.predict(Vector3(0.01, -0.02, 0.005), Vector3(0.05, -0.03, 9.81),
                        0.1);
    batch.processContacts({contact});
    incremental.processContacts({contact});
  }

  EXPECT(assert_equal(navStateFromEstimate(batch.estimate()),
                      navStateFromEstimate(incremental.estimate()), 1e-4));
  const Matrix batchCovariance = batch.navigationStateCovariance();
  const Matrix incrementalCovariance = incremental.navigationStateCovariance();
  CHECK((batchCovariance - incrementalCovariance).norm() /
            std::max(1.0, batchCovariance.norm()) <
        1e-3);
}

/* ************************************************************************* */
TEST(LeggedEstimator, CombinedFixedLagBatchAndIncrementalEndpointsAgree) {
  LeggedEstimatorParams params = makeParams();
  params.imuBias = imuBias::ConstantBias(Vector3(0.002, -0.001, 0.003),
                                         Vector3(0.0001, -0.0002, 0.00015));
  LeggedCombinedFixedLagSmoother batch(
      identityState(), oneFoot(Vector3(1.0, 0.0, 0.0)), I_9x9 * 1e-3, params,
      1.0, {}, LeggedFixedLagEngine::Batch);
  LeggedCombinedFixedLagSmoother incremental(
      identityState(), oneFoot(Vector3(1.0, 0.0, 0.0)), I_9x9 * 1e-3, params,
      1.0, {}, LeggedFixedLagEngine::Incremental);

  const std::array<ContactMeasurement, 3> contacts{
      ContactMeasurement{0, Vector3(1.0, 0.0, 0.0), true},
      ContactMeasurement{0, Vector3(0.99, 0.01, -0.01)},
      ContactMeasurement{0, Vector3(0.98, 0.02, -0.01)}};
  for (const ContactMeasurement& contact : contacts) {
    batch.predict(Vector3(0.01, -0.02, 0.005), Vector3(0.05, -0.03, 9.81), 0.1);
    incremental.predict(Vector3(0.01, -0.02, 0.005), Vector3(0.05, -0.03, 9.81),
                        0.1);
    batch.processContacts({contact});
    incremental.processContacts({contact});
  }

  EXPECT(assert_equal(navStateFromEstimate(batch.estimate()),
                      navStateFromEstimate(incremental.estimate()), 1e-4));
  const Matrix batchCovariance = batch.navigationStateCovariance();
  const Matrix incrementalCovariance = incremental.navigationStateCovariance();
  CHECK((batchCovariance - incrementalCovariance).norm() /
            std::max(1.0, batchCovariance.norm()) <
        1e-3);
}

/* ************************************************************************* */
TEST(LeggedEstimator, FixedLagCovarianceSurvivesMarginalizationAndTouchdown) {
  const LeggedEstimatorParams params = makeParams();
  for (const auto engine :
       {LeggedFixedLagEngine::Batch, LeggedFixedLagEngine::Incremental}) {
    LeggedFixedLagSmoother ordinary(sampleNavState(), fourFootholds(),
                                    I_9x9 * 0.01, params, 0.15, {}, engine);
    LeggedCombinedFixedLagSmoother combined(sampleNavState(), fourFootholds(),
                                            I_9x9 * 0.01, params, 0.15, {},
                                            engine);
    for (int step = 0; step < 12; ++step) {
      const Vector3 omega(0.01, -0.02, 0.005);
      const Vector3 acceleration(0.05, -0.03, 9.81);
      ordinary.predict(omega, acceleration, 0.1);
      combined.predict(omega, acceleration, 0.1);
      std::vector<FootPoseContactMeasurement> contacts;
      if (step % 4 != 2) {
        FootPoseContactMeasurement contact;
        contact.imuPFoot =
            Pose3(Rot3::RzRyRx(0.1, 0.2, -0.1), Point3(0.02, 0.1, -0.48));
        contact.touchdown = step % 4 == 3;
        contact.velocity = FootPoseVelocityMeasurement{};
        contact.velocity->covariance = I_9x9 * 0.01;
        contacts.push_back(contact);
      }
      ordinary.processPoseContacts(contacts);
      combined.processPoseContacts(contacts);

      for (const Matrix& covariance : {ordinary.navigationStateCovariance(),
                                       combined.navigationStateCovariance()}) {
        EXPECT_LONGS_EQUAL(15, covariance.rows());
        EXPECT_LONGS_EQUAL(15, covariance.cols());
        CHECK(covariance.allFinite());
        CHECK(covariance.isApprox(covariance.transpose(), 1e-10));
        CHECK(Eigen::LLT<Matrix>(covariance).info() == Eigen::Success);
        CHECK((covariance.block<6, 3>(0, 6).norm() > 1e-10));
      }
    }

    const Matrix ordinaryPosterior = ordinary.navigationStateCovariance();
    const Matrix combinedPosterior = combined.navigationStateCovariance();
    ordinary.predict(Vector3::Zero(), Vector3(0.0, 0.0, 9.81), 0.01);
    combined.predict(Vector3::Zero(), Vector3(0.0, 0.0, 9.81), 0.01);
    EXPECT(assert_equal(ordinaryPosterior, ordinary.navigationStateCovariance(),
                        1e-12));
    EXPECT(assert_equal(combinedPosterior, combined.navigationStateCovariance(),
                        1e-12));
  }
}

/* ************************************************************************* */
TEST(LeggedEstimator, FixedLagRejectsInvalidEngineAndLag) {
  const LeggedEstimatorParams params = makeParams();
  const auto invalidEngine = static_cast<LeggedFixedLagEngine>(99);
  CHECK_EXCEPTION(LeggedFixedLagSmoother(identityState(), fourFootholds(),
                                         I_9x9, params, 1.0, {}, invalidEngine),
                  std::invalid_argument);
  CHECK_EXCEPTION(
      LeggedCombinedFixedLagSmoother(identityState(), fourFootholds(), I_9x9,
                                     params, 1.0, {}, invalidEngine),
      std::invalid_argument);
  for (const double lag : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                           std::numeric_limits<double>::quiet_NaN()}) {
    CHECK_EXCEPTION(LeggedFixedLagSmoother(identityState(), fourFootholds(),
                                           I_9x9, params, lag),
                    std::invalid_argument);
    CHECK_EXCEPTION(LeggedCombinedFixedLagSmoother(
                        identityState(), fourFootholds(), I_9x9, params, lag),
                    std::invalid_argument);
  }
}

/* ************************************************************************* */
TEST(LeggedEstimator, FixedLagCovarianceUsesNavStateLocalCoordinates) {
  const LeggedEstimatorParams params = makeParams();
  const NavState state = sampleNavState();

  // Anisotropic uncertainty at a rotated pose exposes world/local frame
  // mistakes. Cross blocks also expose a split into independent priors.
  Matrix9 prior = Matrix9::Zero();
  prior.diagonal() << 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09;
  prior(0, 4) = prior(4, 0) = 0.002;
  prior(1, 7) = prior(7, 1) = 0.003;
  prior(3, 6) = prior(6, 3) = 0.004;

  for (const auto engine :
       {LeggedFixedLagEngine::Batch, LeggedFixedLagEngine::Incremental}) {
    LeggedFixedLagSmoother ordinary(state, fourFootholds(), prior, params, 1.0,
                                    {}, engine);
    LeggedCombinedFixedLagSmoother combined(state, fourFootholds(), prior,
                                            params, 1.0, {}, engine);
    Matrix ordinaryExpected = Matrix::Zero(15, 15);
    ordinaryExpected.topLeftCorner<9, 9>() = prior;
    ordinaryExpected.bottomRightCorner<6, 6>() = I_6x6 * 1e-12;

    EXPECT(assert_equal(ordinaryExpected, ordinary.navigationStateCovariance(),
                        1e-10));
    EXPECT(assert_equal(ordinaryExpected, combined.navigationStateCovariance(),
                        1e-10));
  }

  LeggedEstimatorParams waitingParams = params;
  waitingParams.useFullContactInitialization = true;
  LeggedFixedLagSmoother waiting(state, fourFootholds(), prior, waitingParams,
                                 1.0);
  LeggedCombinedFixedLagSmoother combinedWaiting(state, fourFootholds(), prior,
                                                 waitingParams, 1.0);
  CHECK_EXCEPTION(waiting.navigationStateCovariance(), std::logic_error);
  CHECK_EXCEPTION(combinedWaiting.navigationStateCovariance(),
                  std::logic_error);
}

/* ************************************************************************* */
TEST(LeggedEstimator,
     CombinedFullContactInitializationPreservesJointNavigationPrior) {
  LeggedEstimatorParams params = makeRealReplayParams();
  params.useFullContactInitialization = true;
  const NavState state(Rot3::RzRyRx(0.1, 0.25, -0.2), Point3(0.0, 0.0, 0.72),
                       Vector3(0.3, -0.2, 0.1));
  Matrix9 prior = fullContactBaseCovariance(0.2, 0.1);
  prior(1, 7) = prior(7, 1) = 0.002;
  prior(3, 6) = prior(6, 3) = 0.003;

  for (const auto engine :
       {LeggedFixedLagEngine::Batch, LeggedFixedLagEngine::Incremental}) {
    LeggedFixedLagSmoother ordinary(state, fourFootholds(), prior, params, 1.0,
                                    {}, engine);
    LeggedCombinedFixedLagSmoother combined(state, fourFootholds(), prior,
                                            params, 1.0, {}, engine);
    ordinary.processContacts(realReplayContacts(0));
    combined.processContacts(realReplayContacts(0));

    // The NavState-key formulation is the reference for the combined graph's
    // separate pose/velocity keys, including the initialized cross covariance.
    const Matrix expected = ordinary.navigationStateCovariance();
    CHECK((expected.block<6, 3>(0, 6).norm() > 1e-6));
    EXPECT(assert_equal(ordinary.estimate(), combined.estimate(), 1e-8));
    EXPECT(assert_equal(expected, combined.navigationStateCovariance(), 1e-8));
  }
}

/* ************************************************************************* */
TEST(LeggedEstimator, PointVelocityIsRejectedWhileInitializationIsPending) {
  LeggedEstimatorParams params = makeParams();
  params.useFullContactInitialization = true;
  for (size_t variant = 0; variant < 4; ++variant) {
    auto estimator = makeFourPointEstimator(variant, params);
    const ExtendedPose3d prior = estimator->estimate();
    const auto group =
        fourPointContact(correlatedFourPointCovariance(0.0), false, 0.0);
    std::vector<ContactMeasurement> contacts(group.points.begin(),
                                             group.points.end());
    contacts.front().pointVelocity =
        pointVelocityMeasurement(Matrix6::Identity());

    CHECK_EXCEPTION(estimator->processContacts(contacts), std::logic_error);
    EXPECT(assert_equal(prior, estimator->estimate(), 1e-12));
  }
}

/* ************************************************************************* */
TEST(LeggedEstimator, CombinedSmootherDoesNotMutateSharedPreintegrationParams) {
  LeggedEstimatorParams params = makeParams();
  const auto sharedParams = PreintegrationCombinedParams::MakeSharedU(9.81);
  sharedParams->biasAccCovariance = I_3x3 * 0.003;
  sharedParams->biasOmegaCovariance = I_3x3 * 0.004;
  params.preintegrationParams = sharedParams;
  const PreintegrationCombinedParams before = *sharedParams;

  LeggedCombinedFixedLagSmoother first(identityState(), twoFootholds(), I_9x9,
                                       params, 1.0);
  params.biasAccRandomWalkSigma *= 10.0;
  LeggedCombinedFixedLagSmoother second(identityState(), twoFootholds(), I_9x9,
                                        params, 1.0);

  EXPECT(assert_equal(before.biasAccCovariance, sharedParams->biasAccCovariance,
                      1e-12));
  EXPECT(assert_equal(before.biasOmegaCovariance,
                      sharedParams->biasOmegaCovariance, 1e-12));

  params.preintegrationParams.reset();
  CHECK_EXCEPTION(LeggedCombinedFixedLagSmoother(
                      identityState(), twoFootholds(), I_9x9, params, 1.0),
                  std::invalid_argument);
}

/* ************************************************************************* */
TEST(LeggedEstimator, ContactEpisodeKeysDoNotCollideAfterThousandTouchdowns) {
  const LeggedEstimatorParams params = makeParams();
  const Matrix footholds = twoFootholds();
  for (const auto engine :
       {LeggedFixedLagEngine::Batch, LeggedFixedLagEngine::Incremental}) {
    LeggedFixedLagSmoother ordinary(identityState(), footholds, I_9x9 * 0.01,
                                    params, 0.01, {}, engine);
    LeggedCombinedFixedLagSmoother combined(
        identityState(), footholds, I_9x9 * 0.01, params, 0.01, {}, engine);
    const std::array<LeggedEstimator*, 2> estimators{&ordinary, &combined};
    for (LeggedEstimator* estimator : estimators) {
      try {
        // Keep foot 1's first episode active while foot 0 starts new episodes.
        for (int episode = 0; episode <= 1000; ++episode) {
          estimator->predict(Vector3::Zero(), Vector3(0.0, 0.0, 9.81), 0.002);
          estimator->processContacts(
              {ContactMeasurement{0, footholds.col(0), true},
               ContactMeasurement{1, footholds.col(1), false}});
        }
      } catch (const std::exception& error) {
        FAIL(error.what());
      }

      EXPECT(assert_equal(footholds,
                          footholdsFromEstimate(estimator->estimate()), 1e-6));
    }
  }
}

/* ************************************************************************* */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
