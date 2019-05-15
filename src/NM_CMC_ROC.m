function [ CMC, CMCExpectation, nAUCCMC, queryIDs, matchingIDs, mAP ] = NM_CMC_ROC( personIdx, personsIDs, similarityScore, varargin)

p = inputParser;
p.addOptional('ROC', true);
p.addOptional('isDist', false);
p.addOptional('statsFunHandle', @mean);
p.addOptional('useIndexes', false);
p.parse(varargin{:});
opts = p.Results;

% Fix old score size
if size(similarityScore,2) > 1
    similarityScore = similarityScore(:,2);
end

% Probe /gallery
allProbes = personsIDs(:,1);
allGallery = personsIDs(:,2);
if opts.useIndexes
    allProbes = personIdx(:,1);
    allGallery = personIdx(:,2);
end
probe = unique(allProbes);
gallery = unique(allGallery);

% Number of probes / gallery
num_probes = length(probe);
num_gallery = length(gallery);
    
% Probe/ Matching IDs
queryIDs = zeros(1, num_probes);
matchingIDs = zeros(num_probes, num_gallery);

% Loop through all the probe persons
ap = zeros(num_probes,1);
CMC = zeros(num_probes, num_gallery);

% Loop over probes
parfor k=1:num_probes
    
    % Probe ID + similarities with the current probe
    idxk = allProbes(:,1) == probe(k);
    pID = personsIDs(idxk,1);
    pID = pID(1);
    queryIDs(k) = pID;
    simsk  = similarityScore(idxk);

    % Gallery IDs
    gID = personsIDs(idxk,2);

    % Take the final score
    avgScore = simsk;
    if ~opts.useIndexes
        galleryPersonsIDsIterK = unique(gID, 'stable');

        % Compute the score between the probe and every other gallery
        avgScore = arrayfun(@(idbb)( opts.statsFunHandle(simsk(gID==idbb)) ), galleryPersonsIDsIterK);
        gID = galleryPersonsIDsIterK;
    end

    % Dist or score?
    if opts.isDist
        avgScore(isnan(avgScore)) = inf;
        [~, sortedIdx] =  sort(avgScore, 'ascend');
    else
        avgScore(isnan(avgScore)) = -inf;
        [~, sortedIdx] =  sort(avgScore, 'descend');
    end

    % Keep number of gallery fixed..
    toCreate = num_gallery - length(sortedIdx);
    sortedIdx = [sortedIdx; repmat(sortedIdx(end), toCreate,1)];
    
    % Matching pos
    good_index = find(gID==pID);

    % CMC + AP values
    [ap(k), CMC(k,:)] = compute_AP(good_index, [], sortedIdx);

    % Sorted matching IDs
    matchingIDs(k,:) = gID(sortedIdx);

end


% CMC and normalized Area Under Curve (of CMC)
CMC = (sum(CMC, 1) ./ num_probes) * 100;

% Mean average precision (remove any possible NaN)
ap = ap * 100;
ap(isnan(ap)) = 0;
mAP = sum(ap)/length(ap);

% nAUC
nAUCCMC = sum(CMC)./(max(CMC)*num_gallery); 

% Compute CMC expectation
CMCExpectation = NM_expectation_from_CMC(CMC);
end


function [ap, cmc] = compute_AP(good_index, junk_index, index)

cmc = zeros(length(index), 1);
ngood = length(good_index);

old_recall = 0; 
old_precision = 1.0; 
ap = 0; 
intersect_size = 0; 
j = 0; 
good_now = 0; 
njunk = 0;
for n = 1:length(index) 
    flag = 0;
    if ~isempty(find(good_index == index(n), 1)) 
        cmc(n-njunk:end) = 1;
        flag = 1; % good image 
        good_now = good_now+1; 
    end
    if ~isempty(find(junk_index == index(n), 1))
        njunk = njunk + 1;
        continue; % junk image 
    end
    
    if flag == 1%good
        intersect_size = intersect_size + 1; 
    end 
    recall = intersect_size/ngood; 
    precision = intersect_size/(j + 1); 
    ap = ap + (recall - old_recall)*((old_precision+precision)/2); 
    old_recall = recall; 
    old_precision = precision; 
    j = j+1; 
    
    if good_now == ngood 
        return; 
    end 
end 

end

