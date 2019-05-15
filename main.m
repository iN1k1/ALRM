% Author:    Niki Martinel
% Date:      2018/31/07
% Revision:  0.1
% Copyright: Niki Martinel, 2018

%% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%   INITALIZE PARAMETERS
pars = init_parameters(1, 'Market1501', fileparts(which(mfilename)));

%% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@tes@@@@@@@@@@@@@@@@@@@
%   LOAD DATASET
dataset = load_dataset(pars.dataset.name);

%% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%   FEATURE EXTRACTION
features = extract_features( dataset, pars );

%% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
%   GET TRAIN AND TEST SETS
[train, cv, test] = split_dataset(dataset, pars);

%% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
% TRAIN MODEL
model = train_model(features, train, cv, test, pars);

%% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
% EVALUATE MODEL
testRes = eval_model(model, features, test, pars);

%% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
% RESULTS
results = compute_results(dataset, testRes, pars, 'isDist', true, ...
                                                  'useIndexes', true);

%% @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
% SHOW PERFORMANCE
% Plot CMC and mAP
plot(results.CMC(1:min(size(results.CMC,2),100)), 'LineWidth', 3);
xlabel('Rank');
ylabel('Recognition Percentage');
legend(sprintf('mAP = %.2f', results.mAP)); 