function [ model, objective, testRes, trainingTime] = update_metric( model, features, trainSet, testSet, modelPars )
%UPDATE_MODEL Summary of this function goes here
%   Detailed explanation goes here

% Get metric pars and train the initial model
[model, objective, trainingTime] = train_metric(features', model.P, model.K, model.tau, model.lambda, trainSet.index', trainSet.label', modelPars);
% model(c,t).W = zeros(size(model(c,t).W));
% model(c,t).V = zeros(size(model(c,t).W));
% model(c,t).b = 0;


end

