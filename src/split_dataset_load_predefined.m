function [train, cv, test] = split_dataset_load_predefined(dataset, pars)
train = [];
test = [];
cv = [];

switch lower(dataset.name)
    
    case 'market1501'
        main_folder = fullfile(pwd, 'data', 'splits' , 'Market1501');
        
        % Try to load splits
        splitsFile = fullfile(main_folder, 'nm_splits.mat');
        if isfield(pars.dataset, 'allowSameCamera') && pars.dataset.allowSameCamera
            splitsFile = fullfile(main_folder, 'nm_splits_same_camera.mat');
        end            
        if exist(splitsFile, 'file')
            load(splitsFile);
            return;
        end
        
        % Load previously generate correspondance files...
        load(fullfile(main_folder, 'train.mat'))
        load(fullfile(main_folder, 'test.mat'))
        load(fullfile(main_folder, 'query.mat'))
        
        train_set = struct('id', [], 'idx', [], 'cam', []);
        test_set = struct('id', [], 'idx', [], 'cam', []);
        query_set = struct('id', [], 'idx', [], 'cam', []);
        for ii=1:dataset.count
            
            imname = dataset.imageNames{ii};
            inquery = find(strcmpi(imname, query_images)==1,1);
            intrain = find(strcmpi(imname, train_images)==1,1);
            intest = find(strcmpi(imname, test_images)==1, 1);
            if ~isempty(intrain)
                train_set = AddToSet(dataset, train_set, ii);
            elseif ~isempty(intest)
                test_set = AddToSet(dataset, test_set, ii);
            elseif ~isempty(inquery)
                query_set = AddToSet(dataset, query_set, ii);
            end            
        end
        
        % -----------------------------------------------------------------
        % All combinations..
        
        % TRAIN
        allc_train_idx = allcomb(train_set.idx, train_set.idx);
        allc_train_id = allcomb(train_set.id, train_set.id);
        allc_train_cam = allcomb(train_set.cam, train_set.cam);
        % Images of a same person in the same camera ar "junk" => don't use them for training
        if ~isfield(pars.dataset, 'allowSameCamera') || (isfield(pars.dataset, 'allowSameCamera') && ~pars.dataset.allowSameCamera)
            junk = allc_train_id(:,1)==allc_train_id(:,2) & allc_train_cam(:,1) == allc_train_cam(:,2);
            allc_train_idx(junk,:) = [];
            allc_train_id(junk,:) = [];
        end
        %allc_train_cam(junk,:) = [];
        
        % TEST
        allc_test_idx = allcomb(query_set.idx, test_set.idx);
        allc_test_id = allcomb(query_set.id, test_set.id);
        allc_test_cam = allcomb(query_set.cam, test_set.cam);
        % Images of a same person in the same camera ar "junk" => don't use
        % them for evaluation
        if ~isfield(pars.dataset, 'allowSameCamera') || (isfield(pars.dataset, 'allowSameCamera') && ~pars.dataset.allowSameCamera)
            junk = allc_test_id(:,1)==allc_test_id(:,2) & allc_test_cam(:,1) == allc_test_cam(:,2);
            allc_test_idx(junk,:) = [];
            allc_test_id(junk,:) = [];
        end
        %allc_test_cam(junk,:) = [];
        
        % Save
        train.ID = allc_train_id;
        train.index = allc_train_idx;
        train.label = train.ID (:,1) == train.ID (:,2);
        test.ID = allc_test_id;
        test.index = allc_test_idx;
        test.label = test.ID (:,1) == test.ID (:,2);
        
        test.probe = query_set;
        test.gallery = test_set;
        
        save(splitsFile, 'train', 'cv', 'test');
end
end


function [set] = AddToSet(dataset, set, idx)
set.id = [set.id; dataset.personID(idx)];
set.idx = [set.idx; dataset.imageIndex(idx)];
set.cam = [set.cam; dataset.cam(idx)];
end