function testCustomFactorShutdown
% Run in a fresh MATLAB batch process; retain owners through process exit.
% Add the installed toolbox and gtsam_tests to the path before calling this
% function. Success requires the completion marker AND process exit code zero.
% This deliberately leaves registrations alive; do not call from test_gtsam.
assert(gtsam.customFactorRegistry('count') == 0);
scopes = {'all','factorKeys'};
owners = cell(2,5);
for i = 1:numel(scopes)
    values = gtsam.Values;
    values.insert(uint64(1), 2);
    factor = gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(1), uint64(1), ...
        @scalarError, scopes{i});
    graph = gtsam.NonlinearFactorGraph;
    graph.add(factor);
    isam = gtsam.ISAM2;
    isam.update(graph, values);
    copiedIsam = gtsam.ISAM2(isam);
    estimate = copiedIsam.calculateEstimate();
    assert(abs(estimate.atVector(uint64(1))) < 1e-12);
    delete(estimate);
    owners(i,:) = {factor,graph,values,isam,copiedIsam};
end
assignin('base', 'customFactorShutdownOwners', owners);
assert(gtsam.customFactorRegistry('count') == 2);
fprintf('CUSTOM_FACTOR_SHUTDOWN_BODY_COMPLETE REGISTRY=2\n');
end

function [residual,H] = scalarError(~, values)
residual = values.atVector(uint64(1));
H = {1};
end
