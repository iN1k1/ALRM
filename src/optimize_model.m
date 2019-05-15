function [model, objective, trainingTime] = optimize_model(modelInit, X, trainSet, pars)

    % Get labels and indices
    labels = trainSet.label;
    index = trainSet.index;

    %convert labels two -1 / +1
    labels = single(labels);
    labels(labels==0) = -1;

    % Shuffle indexes, labels
    rng(2);
    shuffle = randperm(length(labels), length(labels));
    labels = labels(shuffle);
    index = index(shuffle,:);
    
    % Generate weights
    weights = ones(length(labels),1);
   
    % initialize models pars
    Ls = modelInit.L;
    
    % Value of the objective function after each epoch
    objective = zeros(pars.numEpochs, 1);
    objscore = zeros(pars.numEpochs, 1);
    regul =  zeros(pars.numEpochs, 1);
    
    % Timer for training
    trainingTime = 0;
    
    % Set Nesterov acc par lambda
    lambda = modelInit.lambda;
    
    % Pairwise data samples 
    X_a = X(:, index(:,1));
    X_b = X(:, index(:,2));
   
    
    % Loop for numEpochs
    for s = 1:pars.numEpochs
        
        % Timer
        epochTimer = tic;
        
        % Mini batches
        miniBatchSize = pars.miniBatchSize;
        numIter = ceil(length(labels) / miniBatchSize);
        
        % Get random samples to assing to each mini batch
        allIdx = 1:length(labels);
        miniBatch = cell(numIter, 1);
        if miniBatchSize == 1
            miniBatch = num2cell(allIdx);
        else
            miniBatch = splitvec(randperm(length(allIdx)), numIter);
        end
        
        % =====================================================================
        %       ASVRG
        % =====================================================================

        % Compute the gradients for each sample
        if exist('allGradP', 'var')
            clear allGradP allGradK avgGradP avgGradK Kk Pk;
        end

        % Average gradients with epoch optimal solution
        gLs_all = compute_avg_gradients(X_a, X_b, [], [], weights, labels, Ls, pars);

        % Init batch update matrix
        Lt = Ls;

        % Num iter
        for t=1:numIter

            % Idx in batch
            idx = miniBatch{t};

            % Slice data
            y = labels(idx);
            weight = weights(idx);
            index_a = idx;%index(idx,1);
            index_b = idx;%index(idx,2);

            % Compute the average gradient of the mini batch samples
            % considering the current batch optimal solution
            gLt = compute_avg_gradients(X_a, X_b, index_a, index_b, weight, y, Lt, pars);
            gLs = compute_avg_gradients(X_a, X_b, index_a, index_b, weight, y, Ls, pars);

            % 1. Compute the average gradient of the mini batch samples
            %    considering the current global optimal solution
            % 2. Get fused gradients
            % 3. Update the gradients and perform proximal mapping
            % 4. Apply Nesterov acceleration
            gLt_mb = gLt - gLs + gLs_all;
            tmp = prox_mapping(gradient_step(Lt, gLt_mb, pars.eta), pars.alpha, pars.eta);
            Lt = tmp + lambda * (tmp - Lt);

        end
            
        % Update L
        Ls = Lt;
        
        % Update time..
        trainingTime = trainingTime + toc(epochTimer);
        
        % Evalaute the objective function
        [objective(s), objscore(s), regul(s)] = objective_function_evaluation(X_a, X_b, 1:length(labels), 1:length(labels), weights, labels, Ls, pars);
        
        % Display
        if pars.verbose
            fprintf('\nObjective at epoch %d = %.8f + %.8f (epoch %.2f s)', s, objscore(s), regul(s), trainingTime/s);
        end
        
        % Do we have to stop?
        if s > 2 && abs(objective(s-1) - objective(s)) < pars.stopObjDiff
            if pars.verbose
                fprintf('\nStop Optimization: difference between consecutive objective values less than the given threshold');
            end
            break;
        end
    end
    
    
    % Output model structure
    model.L = Ls;
    model.lambda = lambda;
    
    % Training time
    trainingTime = trainingTime / pars.numEpochs;
    
    if pars.verbose, fprintf('\n'); end
    
end

function [score] = compute_score(y, LX_a, LX_b)
dis = 0.5 * sum( (LX_a-LX_b).^2,1)';
score = y .* ( - dis );
end

function [g] = gradient_step(g, gGrad, eta)
g = g - (eta * gGrad);
end

function [proxP] = prox_mapping(Phat, alpha, eta)
proxP = soft_thresholding(Phat, alpha, eta);
end

function [Mth] = soft_thresholding(M, alpha, eta)
Mth = zeros(size(M), 'like', M);
mult = alpha * eta;
for i=1:size(M,1)
    Mth(i,:) = M(i,:) * max(0, 1 - (mult / (norm(M(i,:), 2))));
end
end

function [avgGradP] = compute_avg_gradients(X_a, X_b, index_a, index_b, weights, labels, L, pars)

% Compute projections
if isempty(index_a) || isempty(index_b)
    featDiff = X_a - X_b;
    LX_a = L * X_a;
    LX_b = L * X_b;
else
    featDiff = X_a(:, index_a) - X_b(:, index_b);
    LX_a = L * X_a(:, index_a);
    LX_b = L * X_b(:, index_b);
end

featDiff_P = L * featDiff;

% Scores
score = compute_score(labels, LX_a, LX_b);

% Average gradient
avgGradP = gradient_smooth_hinge(score, weights, labels, featDiff_P, featDiff);
avgGradP = avgGradP / length(labels);

end

function [objective, loss, regul] = objective_function_evaluation(X_unique_a, X_unique_b, ...
    index_a, index_b, weights, labels, L, pars)
    
% Get features
LX_a = L * X_unique_a;
LX_b = L * X_unique_b;

% Compute the score for each pair
scores = compute_score(labels, LX_a(:, index_a), LX_b(:, index_b));

% Apply smooth hinge loss
loss = smooth_hinge_loss(scores);

% Average objective score +  sum of l21 norms
loss = mean(weights .* loss);
regul = pars.alpha * NM_norm(gather(L), '21');
objective = loss + regul;
end

