function testCustomFactorValuesScope
% Exercise owning callback snapshots, failures, and graph/optimizer lifetimes.
% Run with the matching installed toolbox and this test directory on the path.
baseline = gtsam.customFactorRegistry('count');
testScopesAndTypes;
testVectorResidual;
testRetainedSnapshots;
testFailures;
testOptimizerCopies;
assert(gtsam.customFactorRegistry('count') == baseline);
fprintf('CustomFactor values scope tests passed.\n');
end

function testScopesAndTypes
% MATLAB numeric scalars use the existing one-element-vector binding.
keys = [gtsam.symbol('p', 9), gtsam.symbol('b', 3), gtsam.symbol('v', 1)];
values = gtsam.Values;
values.insert(keys(1), gtsam.Pose3(gtsam.Rot3.RzRyRx(0.2,-0.1,0.3), [1;2;3]));
values.insert(keys(2), 2);
values.insert(keys(3), [3;4]);
values.insert(uint64(99), 100);
model = gtsam.noiseModel.Unit.Create(1);
full = gtsam.CustomFactor(model, keys, @mixedError);
explicitFull = gtsam.CustomFactor(model, keys, @mixedError, "all");
subset = gtsam.CustomFactor(model, keys, @mixedError, 'factorKeys');
assert(full.unwhitenedError(values) == 10);
assert(subset.unwhitenedError(values) == full.unwhitenedError(values));
assert(explicitFull.error(values) == full.error(values));
[A,b] = full.linearize(values).jacobian();
[subsetA,subsetB] = subset.linearize(values).jacobian();
assert(isequal(A,subsetA) && isequal(b,subsetB));
R = values.atPose3(keys(1)).rotation().matrix();
assert(norm(A - [zeros(1,3), R(1,:), 1, 1, 1]) < 1e-12);
assert(isequal(subset.keys().at(0), keys(1)));
assert(isequal(subset.keys().at(1), keys(2)));

% Unrelated values remain available by default and are excluded by factorKeys.
inspectAll = gtsam.CustomFactor(model, keys(2), @scopeError);
inspectSubset = gtsam.CustomFactor(model, keys(2), @scopeError, "factorKeys");
assert(inspectAll.unwhitenedError(values) == 4);
assert(inspectSubset.unwhitenedError(values) == 1);

% Duplicate declared keys do not require duplicate Values entries.
duplicate = gtsam.CustomFactor(model, [keys(2),keys(2)], @scopeError, 'factorKeys');
assert(duplicate.unwhitenedError(values) == 1);
emptyKeys = gtsam.CustomFactor(model, uint64([]), @scopeError, 'factorKeys');
assert(emptyKeys.unwhitenedError(values) == 0);
delete(emptyKeys); delete(duplicate); delete(inspectAll); delete(inspectSubset);
delete(full); delete(explicitFull); delete(subset); delete(values);
end

function [residual,H] = mixedError(this, values)
keys = this.keys();
pose = values.atPose3(keys.at(0));
scalar = values.atVector(keys.at(1));
vector = values.atVector(keys.at(2));
translation = pose.translation();
rotation = pose.rotation().matrix();
residual = translation(1) + scalar + sum(vector);
H = {[zeros(1,3),rotation(1,:)], 1, ones(1,2)};
end

function residual = scopeError(~, values)
residual = double(values.size());
end

function testVectorResidual
% Exercise multiple residual rows, a non-square Jacobian, and KeyVector input.
key = gtsam.symbol('v', uint64(91));
keys = gtsam.KeyVector;
keys.push_back(key);
values = gtsam.Values;
values.insert(key, [1;2;3]);
values.insert(uint64(99), [5;6]);
model = gtsam.noiseModel.Unit.Create(2);
for scope = {'all','factorKeys'}
    factor = gtsam.CustomFactor(model, keys, @vectorResidual, scope{1});
    assert(isequal(factor.unwhitenedError(values), [7;8]));
    linear = factor.linearize(values);
    [A,b] = linear.jacobian();
    assert(isequal(A, [1,0,2;0,1,2]) && isequal(b, [-7;-8]));
    delete(linear);
    delete(factor);
end
delete(values);
delete(keys);
end

function [residual,H] = vectorResidual(this, values)
vector = values.atVector(this.keys().at(0));
residual = [vector(1)+2*vector(3);vector(2)+2*vector(3)];
H = {[1,0,2;0,1,2]};
end

function testRetainedSnapshots
for scope = {'all','factorKeys'}
    retained = containers.Map('KeyType','double','ValueType','any');
    values = gtsam.Values;
    values.insert(uint64(1), [1;2]);
    values.insert(uint64(2), 5);
    factor = gtsam.CustomFactor(gtsam.noiseModel.Unit.Create(1), uint64(1), ...
        @(~,snapshot) retainValues(snapshot,retained), scope{1});
    assert(factor.unwhitenedError(values) == 3);
    values.update(uint64(1), [4;5]);
    assert(factor.unwhitenedError(values) == 9);
    first = retained(1); second = retained(2);
    first.update(uint64(1), [-1;-2]);
    assert(isequal(values.atVector(uint64(1)), [4;5]));
    delete(values); delete(factor);
    assert(isequal(first.atVector(uint64(1)), [-1;-2]));
    assert(isequal(second.atVector(uint64(1)), [4;5]));
    delete(first); delete(second);
end
end

function residual = retainValues(values, retained)
% containers.Map is a handle; this assignment updates the caller's map.
retained(retained.Count+1) = values; %#ok<NASGU>
residual = sum(values.atVector(uint64(1)));
end

function testFailures
model = gtsam.noiseModel.Unit.Create(1);
baseline = gtsam.customFactorRegistry('count');
for scope = {'bad', '', ["all","factorKeys"], 0, {}}
    assertError(@() gtsam.CustomFactor(model, uint64(1), @scopeError, scope{1}), ...
        'gtsam:CustomFactor:invalidValuesScope');
end
assertError(@() gtsam.CustomFactor([], uint64(1), @scopeError, 'factorKeys'), '');
assert(gtsam.customFactorRegistry('count') == baseline);
values = gtsam.Values;
values.insert(uint64(1), 2);
for scope = {'all','factorKeys'}
    missing = gtsam.CustomFactor(model, uint64(99), @scalarError, scope{1});
    assertError(@() missing.unwhitenedError(values), 'missingKey');
    assertError(@() missing.linearize(values), 'missingKey');
    delete(missing);
    throwing = gtsam.CustomFactor(model, uint64(1), @throwingError, scope{1});
    for repeat=1:5
        assertError(@() throwing.error(values), 'CustomFactorTest:callbackFailure');
        assertError(@() throwing.linearize(values), 'CustomFactorTest:callbackFailure');
    end
    % Errors must also unwind through native optimizer callers safely.
    graph = gtsam.NonlinearFactorGraph; graph.add(throwing);
    assertError(@() gtsam.LevenbergMarquardtOptimizer(graph,values), ...
        'CustomFactorTest:callbackFailure');
    isam = gtsam.ISAM2;
    assertError(@() isam.update(graph,values), 'CustomFactorTest:callbackFailure');
    delete(isam); delete(graph);
    delete(throwing);
    for residual = {[1;2], [1,2], single(1), sparse(1), 1i}
        bad = gtsam.CustomFactor(model, uint64(1), @(~,~) residual{1}, scope{1});
        assertError(@() bad.error(values), 'gtsam:CustomFactor:invalidResidual');
        delete(bad);
    end
    badBlocks = {1, {}, {ones(2,1)}, {ones(1,2)}, {single(1)}, {sparse(1)}, {1i}, {[]}};
    ids = {'invalidJacobians','invalidJacobianCount','invalidJacobians', ...
           'invalidJacobians','invalidJacobians','invalidJacobians', ...
           'invalidJacobians','invalidJacobians'};
    for i=1:numel(badBlocks)
        bad = gtsam.CustomFactor(model, uint64(1), ...
            @(~,~) badJacobian(badBlocks{i}), scope{1});
        assertError(@() bad.linearize(values), ['gtsam:CustomFactor:',ids{i}]);
        delete(bad);
    end
    % A successful wrapper call after errors also checks restored stream state.
    good = gtsam.CustomFactor(model, uint64(1), @scalarError, scope{1});
    assert(good.error(values) == 2);
    assert(good.linearize(values).jacobian() == 1);
    values.print('Values after callback failures: ');
    delete(good);
end
delete(values);
assert(gtsam.customFactorRegistry('count') == baseline);
end

function [residual,H] = scalarError(this, values)
residual = values.atVector(this.keys().at(0));
H = {1};
end

function [residual,H] = throwingError(~, ~) %#ok<STOUT>
% This callback intentionally never returns either output.
error('CustomFactorTest:callbackFailure', 'Intentional callback failure.');
end

function [residual,H] = badJacobian(blocks)
residual = 1;
H = blocks;
end

function assertError(callback, identifier)
try
    callback();
catch exception
    if strcmp(identifier, 'gtsam:CustomFactor:invalidValuesScope')
        assert(strcmp(exception.identifier, identifier), exception.message);
    elseif strcmp(identifier, 'missingKey')
        assert(contains(exception.message, 'does not exist'), exception.message);
    elseif ~isempty(identifier)
        % Native exceptions pass through the wrapper's generic C++ boundary.
        assert(isempty(exception.identifier), exception.message);
        assert(contains(exception.message, ['[',identifier,']']), exception.message);
    end
    if strcmp(identifier, 'CustomFactorTest:callbackFailure')
        assert(contains(exception.message, 'throwingError'));
    end
    return
end
error('CustomFactorTest:expectedFailure', 'Expected callback to fail.');
end

function testOptimizerCopies
baseline = gtsam.customFactorRegistry('count');
model = gtsam.noiseModel.Unit.Create(1);
for scope = {'all','factorKeys'}
    for repeat=1:10
        factor = gtsam.CustomFactor(model, uint64(1), @scalarError, scope{1});
        graph = gtsam.NonlinearFactorGraph;
        graph.add(factor);
        copiedGraph = gtsam.NonlinearFactorGraph(graph);
        delete(graph);
        values = gtsam.Values;
        values.insert(uint64(1), 2);
        optimizer = gtsam.LevenbergMarquardtOptimizer(copiedGraph,values);
        result = optimizer.optimize();
        assert(abs(result.atVector(uint64(1))) < 1e-8);
        isam = gtsam.ISAM2;
        isam.update(copiedGraph,values);
        copiedIsam = gtsam.ISAM2(isam);
        delete(isam); delete(optimizer); delete(copiedGraph);
        copiedIsam.update();
        estimate = copiedIsam.calculateEstimate();
        assert(abs(estimate.atVector(uint64(1))) < 1e-8);
        delete(estimate); delete(result); delete(copiedIsam); delete(factor); delete(values);
        assert(gtsam.customFactorRegistry('count') == baseline);
    end
    % Explicit early deletion still invalidates the registration. It must fail
    % as a MATLAB exception, rather than accessing a destroyed callback handle.
    values = gtsam.Values; values.insert(uint64(1), 2);
    factor = gtsam.CustomFactor(model, uint64(1), @scalarError, scope{1});
    graph = gtsam.NonlinearFactorGraph; graph.add(factor);
    delete(factor);
    assertError(@() graph.error(values), '');
    delete(graph); delete(values);
    assert(gtsam.customFactorRegistry('count') == baseline);
end
end
