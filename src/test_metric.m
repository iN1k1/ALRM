function [ S, probs, idx_a, idx_b, id_a, id_b] = test_metric( model, X, ID, index, modelPars, bComputeAllPairs )

% Pre-Compute projections
featProj_L = model.L*X;
%featProj_L = X;

% Now, do we have to compute it for all possible pairs or only for the ones
% in the index..
if bComputeAllPairs

    [idx_a, ip] = unique(index(:,1));
    [idx_b, ig] = unique(index(:,2));
    id_a = ID(ip,1);
    id_b = ID(ig,2);

    LX_a = featProj_L(:,idx_a);
    LX_b = featProj_L(:,idx_b);
    
    S = zeros(length(idx_a), length(idx_b));
    for i =1:length(idx_a)
        diff = bsxfun(@minus, LX_b, LX_a(:,i));
        diff = arrayfun(@(idx)(norm(diff(:,idx))), 1:size(diff,2));
        diff = 0.5 * (diff .^2);
        %D(i, :) = tau - diff;
        S(i, :) = diff;
    end
else
    idx_a = index(:,1);
    idx_b = index(:,2);
    id_a = ID(:,1);
    id_b = ID(:,2);
    S = zeros(size(index,1), 1);
    parfor ii=1:size(index,1)
        S(ii) = norm(featProj_L(:,idx_a(ii))-featProj_L(:,idx_b(ii)))^2;
    end
end

% Probabilities using Platt scaling
probs = zeros(size(S));
if isfield(model, 'plattA')
    probs = 1 ./ ( 1 + exp(model.plattA*S + model.plattB) );
end

end