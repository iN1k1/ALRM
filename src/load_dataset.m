function [dataset] = load_dataset(datasetName)
load(fullfile(pwd, 'data', ['dataset_' datasetName]));
end