function [outputStructure] = train_model(features, trainSet, crossValidationSet, testSet, pars)


fprintf('Training model...');
trainTime = tic;

% Try to load data
trainFile = fullfile(pars.settings.dataFolder, [pars.settings.outputPrefix '_model.mat']);
if exist(trainFile, 'file')
    load(trainFile);
else

    % Available features
    availableFeatures = fieldnames(pars.features);
    availableFeatures = availableFeatures(structfun(@(x)(x.enabled), pars.features));
    features = single(features)';

    % Center features (zero mean)
    X = bsxfun(@minus, features, mean(features, 1))';

    % for each camera pair
    for c=1:size(pars.settings.testCams,1)

        % # Tests
        for t=1:pars.settings.numTests

            % Init model
            model(c,t) = init_model(X, availableFeatures{1}, pars);

            % Train  and evaluate model
            [model(c,t), objective(c,t,:), trainingTime(c,t)] = optimize_model(model(c,t), X, trainSet(c,t), pars.classifier);
        end 
    end

    % Combine all data in the output structure
    outputStructure.model = model;
    outputStructure.objective = objective;
    outputStructure.trainingTime = trainingTime;

    try
        save(trainFile, 'outputStructure');
    catch ME
        warning('train:saving_error', 'Unable to save training model on file %s.', trainFile)
    end
end

% Training time
fprintf('done in %.2f(s)\n', toc(trainTime));



end