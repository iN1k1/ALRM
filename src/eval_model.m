function [ testRes ] = eval_model( model, X, test, pars )

fprintf('Evaluating model...');
ttt = tic;

testRes = repmat(struct('score', [], 'ID', [], 'index', []), 1, length(model));

% Center features (zero mean)
X = single(X)';
mu = mean(X, 1);
X = bsxfun(@minus, X, mu)';

% Run separate tests for each trained model
for t=1:length(model)
    index = test(t).index;
    id = test(t).ID;
    [score, probs] = test_metric(model.model(t), X, test(t).ID, test(t).index, pars.classifier, false);
    testRes(t).score = score;
    testRes(t).ID = id;
    testRes(t).index = index;
end

fprintf('done in %.2f(s)\n', toc(ttt));


end

