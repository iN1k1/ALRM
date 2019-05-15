function [results] = compute_results( dataset, tests, pars, varargin )


% Parse optional input parameters
p = inputParser;
p.addOptional('isDist', false);
p.addOptional('loadExisting', true);
p.addOptional('saveSuffix', '');
p.addOptional('useIndexes', false);
p.parse(varargin{:});
opts = p.Results;

fprintf('Computing results...');
ttt = tic;


% Try to load data
resultsFile = fullfile(pars.settings.dataFolder, sprintf('%s_results%s.mat', pars.settings.outputPrefix, opts.saveSuffix));
if opts.loadExisting && exist(resultsFile, 'file')
    load(resultsFile);
else

    % Loop through all tests
    for t=1:size(tests,2)
        [CMC{t}, CMCExpectation(:, t), nAUCCMC(t), ...
            queryPersonsIDs{t}, matchingIDs{t}, ap(t)] ...
                    = NM_CMC_ROC(tests(1,t).index, tests(1,t).ID, tests(1,t).score, varargin{:});
    end

    % Average values    
    results.CMC_all = vertcat(CMC{:});
    results.CMC = mean(results.CMC_all,1);
    results.CMCmed = median(results.CMC_all,1);
    results.CMCstd = std(results.CMC_all,[],1);

    results.CMCExpectation = mean(CMCExpectation, 2);
    results.nAUCCMC = mean(nAUCCMC); 
    results.nAUCCMCmed = sum(results.CMCmed)/(100*length(results.CMCmed));

    % mAP
    results.mAP = mean(ap);

    % Best run
    [~, bestRunIdx] = max(nAUCCMC);
    results.CMCBest = CMC{bestRunIdx};
    results.nAUCCMCBest = nAUCCMC(bestRunIdx);
    results.CMCExpectationBest = CMCExpectation(:,bestRunIdx);

    % Matching results
    results.queryPersonsIDs = queryPersonsIDs;
    results.matchingIDs = matchingIDs;

    % Save data
    if opts.loadExisting
        try
            save(resultsFile, 'results');
        catch ME
            warning('compute_results:save', 'Unable to save results data on file %s.', resultsFile)
        end
    end
end
fprintf('done in %.2f(s)\n', toc(ttt));

end