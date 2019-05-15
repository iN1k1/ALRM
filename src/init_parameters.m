function [pars] = init_parameters(testID, datasetName, folder)

fprintf('Initialize parameters...');
t = tic;


%% ========================================================================
%   MAIN SETTINGS
% =========================================================================

% Test ID
pars.settings.testID = sprintf('%03d', testID);
    
% dataset parameters
pars.dataset.name = datasetName;
pars.dataset.imageColorSpace = 'RGB';

% image sizes
pars.dataset.imageWidth         = 64;
pars.dataset.imageHeight        = 128;
pars.dataset.imageMagFactor     = 1;
pars.dataset.useMasks           = true;  
pars.dataset.loadCams           = true;    

pars.settings.testCams = [1 2];
pars.settings.numTests = 1;
pars.settings.numPersons = [];
pars.settings.numImages = 225000; % -1 / 225000 (only for Market.. this is the number of negative samples to keep)
pars.settings.useSameImageIndexForPositiveAndNegativeSamples = true;
pars.settings.numberSamplesPerPersonTraining = [pars.settings.numImages pars.settings.numImages];
pars.settings.numberSamplesPerPersonTesting = [pars.settings.numImages pars.settings.numImages];
pars.settings.trainAndTestWithSamePersons = false;
pars.settings.testPeopleIDs = [];
pars.settings.learningSets = [0.5 0.5];
pars.settings.kfold = [];

% Fix random generator seed
rng(2);

%% ========================================================================
%   FEATURES
% =========================================================================

%----------------------------------------------------------------------
% LOMO
pars.features.lomo.enabled = false;
pars.features.lomo.colorSpace = 'RGB';
pars.features.lomo.dim = 200;

%----------------------------------------------------------------------
% IDE
pars.features.ide.enabled = true;
pars.features.ide.colorSpace = 'RGB';
pars.features.ide.dim = 400;


%% ========================================================================
%   CLASSIFIER/METRIC
% =========================================================================
pars.classifier.method = 'lrsml';
pars.classifier.dim = -1;
pars.classifier.doWhitening = false;
pars.classifier.pcainit = true;
pars.classifier.optim = 'ASVRG'; 
pars.classifier.eta = 0.005; % 0.7 viper/prid - 0.01 market1501; 
pars.classifier.alpha = 1e-7; % 1e-7 viper
pars.classifier.numEpochs = 5;
pars.classifier.miniBatchSize = 24; %24
pars.classifier.kfold = 0;
pars.classifier.verbose = true;
pars.classifier.stopObjDiff = 0.00005;

% Update final dataset image size
pars.dataset.imageWidth   = pars.dataset.imageWidth  * pars.dataset.imageMagFactor;
pars.dataset.imageHeight  = pars.dataset.imageHeight * pars.dataset.imageMagFactor;
      
% Output file on which save test data
pars.settings.outputPrefix = [pars.dataset.name, '_Id', pars.settings.testID];
pars.settings.rootFolder = folder;
pars.settings.dataFolder = fullfile(pars.settings.rootFolder, pars.settings.outputPrefix);
pars.settings.resultsFolder = fullfile(pars.settings.rootFolder, 'results');

if ~exist(fullfile(pars.settings.dataFolder), 'file')
    mkdir(fullfile(pars.settings.dataFolder));
end


% Save parameters structure
paramsFile = fullfile(pars.settings.dataFolder, [pars.settings.outputPrefix '_params.mat']);
save(paramsFile, 'pars');

fprintf('done in %.2f(s)\n', toc(t));
end