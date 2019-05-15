function [ extendedDataset ] = extend_dataset( dataset, pars )
%EXTEND_DATASET Summary of this function goes here
%   Detailed explanation goes here

cams = unique(dataset.cam);

noiseType = [1 2 3 4];
flipType = [1 2];
alterType = [1 2];
modifTypes = allcomb(noiseType, flipType, alterType);
allEffects = modifTypes(randperm(size(modifTypes,1)), :);
         
extendedDataset.images = [];
extendedDataset.masks = [];
extendedDataset.imageNames = {};
extendedDataset.cam = [];
extendedDataset.personID = [];
extendedDataset.personSubsetImageIndex = [];
extendedDataset.imageIndex = [0];
extendedDataset.peopleCount = dataset.peopleCount;
extendedDataset.name = dataset.name;

newImagesPerPerson = pars.settings.extendNumberOfImagesPerPersonPerCamera;
if isempty(newImagesPerPerson) || newImagesPerPerson == 0
    extendedDataset = dataset;
    else
    for p=1:dataset.peopleCount
        for cam = cams
            indexesForPerson = dataset.imageIndex(dataset.personID == p & dataset.cam == cam);
            newImages = zeros(size(dataset.images,1), size(dataset.images,2), size(dataset.images,3), pars.settings.extendNumberOfImagesPerPersonPerCamera, 'uint8');
            newMask = zeros(size(dataset.masks,1), size(dataset.masks,2), pars.settings.extendNumberOfImagesPerPersonPerCamera, 'double');

            for e=1:newImagesPerPerson
                idx = randperm(length(indexesForPerson));
                image = im2double(dataset.images(:,:,:, indexesForPerson(idx)));
                mask = dataset.masks(:,:,indexesForPerson(idx));

                effects = allEffects(e,:);
                effectsOrder = randperm(size(effects,2));

                % Create new image
                for eff = effectsOrder
                    typeID = effects(eff);
                    switch eff
                        case 1
                            image = add_noise(image, typeID);
                        case 2
                            [image, mask] = flip_image(image, mask, typeID);
                        case 3
                            image = alter_brightness(image, typeID);
                    end
                end

                newImages(:,:,:,e) = im2uint8(image);
                newMask(:,:,e) = mask;

            end

            % add data to original dataset
            extendedDataset.images          = cat(4, extendedDataset.images, dataset.images(:,:,:,indexesForPerson), newImages);
            extendedDataset.masks           = cat(3, extendedDataset.masks, dataset.masks(:,:,indexesForPerson), newMask);
            extendedDataset.imageNames      = [extendedDataset.imageNames repmat(dataset.imageNames(indexesForPerson), 1, newImagesPerPerson+1 )];
            extendedDataset.cam             = [extendedDataset.cam repmat(dataset.cam(indexesForPerson(1)), 1, newImagesPerPerson+1)];
            extendedDataset.personID        = [extendedDataset.personID repmat(dataset.personID(indexesForPerson(1)), 1, newImagesPerPerson+1)];
            extendedDataset.personSubsetImageIndex = [extendedDataset.personSubsetImageIndex 1:(newImagesPerPerson+1)];
        end
    end

    extendedDataset.imageIndex = 1:length(extendedDataset.cam);
    extendedDataset.count = length(extendedDataset.cam);
end

end

function [image] = add_noise(image, typeID)
switch typeID
    case 1
        mu = 0;
        sigma = 0.001;
        image = imnoise(image, 'gaussian', mu, sigma);
        
    case 2
        image = imnoise(image, 'poisson');
    case 3
        image = imnoise(image, 'salt & pepper', 0.005);
    case 4
        image = imnoise(image, 'speckle', 0.001);
end

end

function [image, mask] = flip_image(image, mask, typeID)
    switch typeID
        
        % Flip columns/ horizontal flip
        case 1
            image = flipdim(image, 2);
            mask = flipdim(mask,2);
        case 2
            % Noflip
    end
end

function [image] = alter_brightness(image, typeID)
minR = min(min(image(:,:,1)));
minG = min(min(image(:,:,2)));
minB = min(min(image(:,:,3)));
maxR = max(max(image(:,:,1)));
maxG = max(max(image(:,:,2)));
maxB = max(max(image(:,:,3)));
factor = 0;
switch typeID
    case 1
        % Make image darker
        factor = -0.001;
    case 2
        % Make image brighter
        factor = 0.001;
end
if maxR+factor > 1, maxR = 1-factor; end
if maxG+factor > 1, maxG = 1-factor; end
if maxB+factor > 1, maxB = 1-factor; end

image = imadjust(image,[minR minG minB; maxR+factor maxG+factor maxB+factor],[]);

    
end
