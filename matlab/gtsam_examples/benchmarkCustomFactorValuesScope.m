function results = benchmarkCustomFactorValuesScope(valueCounts, factorCount, repetitions)
% Compare identical one-dimensional Gauss-Markov graphs with full and factor-key snapshots.
% Example: results = benchmarkCustomFactorValuesScope([501 5000 20000],500,7);
% Construction and correctness checks are outside timings. Raw samples are
% returned for saving by the caller; this benchmark does not run in test_gtsam.
if nargin < 1, valueCounts = [501 5000 20000]; end
if nargin < 2, factorCount = 500; end
if nargin < 3, repetitions = 7; end
validateattributes(factorCount, {'double'}, {'scalar','integer','positive'});
validateattributes(valueCounts, {'double'}, {'vector','integer','>=',factorCount+1});
validateattributes(repetitions, {'double'}, {'scalar','integer','positive'});
% MATLAB inserts numeric scalars as one-element vectors, accessed via atVector.
baseline = gtsam.customFactorRegistry('count');
phi = exp(-30/120);
noise = gtsam.noiseModel.Isotropic.Sigma(1, 2*sqrt(-expm1(-2*30/120)));
graphs = {gtsam.NonlinearFactorGraph, gtsam.NonlinearFactorGraph};
scopes = {'all','factorKeys'};
factors = cell(factorCount,2);
for i=1:factorCount
    previous = uint64(i); current = uint64(i+1);
    callback = @(~,values) biasError(values,previous,current,phi);
    for mode=1:2
        factors{i,mode} = gtsam.CustomFactor(noise,[previous,current],callback,scopes{mode});
        graphs{mode}.add(factors{i,mode});
    end
end
results.valueCounts = valueCounts;
results.factorCount = factorCount;
results.repetitions = repetitions;
results.columns = {'allError','factorKeysError','allLinearize','factorKeysLinearize'};
results.seconds = zeros(numel(valueCounts),repetitions,4);
for n=1:numel(valueCounts)
    values = gtsam.Values;
    for j=1:valueCounts(n), values.insert(uint64(j),sin(j)); end
    assert(graphs{1}.error(values) == graphs{2}.error(values));
    linearFull = graphs{1}.linearize(values);
    linearSubset = graphs{2}.linearize(values);
    [A,b] = linearFull.jacobian(); [subsetA,subsetB] = linearSubset.jacobian();
    assert(isequal(A,subsetA) && isequal(b,subsetB));
    delete(linearFull); delete(linearSubset);
    for warm=1:2
        for mode=1:2
            graphs{mode}.error(values);
            linear = graphs{mode}.linearize(values); delete(linear);
        end
    end
    for repeat=1:repetitions
        order = [1,2];
        if mod(repeat,2) == 0, order = [2,1]; end
        for mode=order
            timer = tic; graphs{mode}.error(values);
            results.seconds(n,repeat,mode) = toc(timer);
            timer = tic; linear = graphs{mode}.linearize(values);
            results.seconds(n,repeat,mode+2) = toc(timer);
            delete(linear);
        end
    end
    fprintf('N=%d F=%d all_error=%.9f subset_error=%.9f all_linearize=%.9f subset_linearize=%.9f\n', ...
        valueCounts(n),factorCount,squeeze(median(results.seconds(n,:,:),2)));
    delete(values);
end
for mode=1:2
    delete(graphs{mode});
    for i=1:factorCount, delete(factors{i,mode}); end
end
assert(gtsam.customFactorRegistry('count') == baseline);
end

function [residual,H] = biasError(values,previous,current,phi)
residual = values.atVector(current) - phi*values.atVector(previous);
H = {-phi,1};
end
