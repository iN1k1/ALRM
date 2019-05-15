function [ features ] = extract_features( dataset, pars )

fprintf('Get features...');
t = tic;

fname = sprintf('LOMO_%s.mat', dataset.name);
if  pars.features.ide.enabled
    fname = sprintf('IDE_%s.mat', dataset.name);
end

fpath = fullfile(pwd, 'data', fname);
if exist(fpath, 'file')
    load(fpath);
else
    % Compute lomo features
    features = LOMO(dataset.images);
    save(fpath, 'features');
end

if pars.features.ide.enabled
    features = double(features);
    sum_val = sqrt(sum(features.^2));
    for n = 1:size(features, 1)
        features(n, :) = features(n, :)./sum_val;
    end
end

fprintf('done in %.2f(s)\n', toc(t));


end

