function [model] = init_model(X, featureType, pars)

% Matrix proj dim = r
dim = pars.classifier.dim;
if dim == -1
    dim = pars.features.(featureType).dim;
end

% PCA
if pars.classifier.pcainit
    
    % Try to load file
    fname = ['PCA_' upper(featureType) '_' pars.dataset.name];
    fpath = fullfile(pwd, 'data', fname);
    try 
        load(fpath);
    catch    
        [covEigvec, ~, covEigval] = pca(X', 'Algorithm','svd','Economy',true);
        save(fpath,  'covEigvec', 'covEigval');
    end

    % PCA dim
    if ~pars.classifier.doWhitening
        % PCA
        LInit = covEigvec(:, 1:dim)';
    else
        % PCA + whitening
        LInit = diag(1 ./ sqrt(covEigval(1:dim) + 1e-5)) * covEigvec(:, 1:dim)';
    end
else
    LInit = randn(dim, size(X,2));
end

% Initialize model pars
model.L = LInit;
model.lambda = (pars.classifier.miniBatchSize-2) / (pars.classifier.miniBatchSize+2);

end