function [X, Y] = loadCOIL20(dataPath)
    imageFiles = dir(fullfile(dataPath, '*.png'));
    numImages = length(imageFiles);
    
    if numImages == 0
        error('Please check the datapath: %s', dataPath);
    end

    sampleImg = imread(fullfile(imageFiles(1).folder, imageFiles(1).name));
    imgSize = size(sampleImg);
    numPixels = imgSize(1) * imgSize(2);

    X = zeros(numPixels, numImages);
    Y = zeros(1, numImages);

    for i = 1:numImages
        img = imread(fullfile(imageFiles(i).folder, imageFiles(i).name));
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        X(:, i) = double(img(:));
        fileName = imageFiles(i).name;
        objID = sscanf(fileName, 'obj%d');
        Y(i) = objID;
    end

    X = X / 255;
end
