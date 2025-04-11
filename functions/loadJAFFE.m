function [X, X_re, Y, labelNames] = loadJAFFE(dataPath)
    % X_re here represents the resized picture
    labelNames = {'AN', 'DI', 'FE', 'HA', 'NE', 'SA', 'SU'};

    imgFiles = dir(fullfile(dataPath, '*.tiff'));
    numImages = length(imgFiles);

    sampleImg = imread(fullfile(imgFiles(1).folder, imgFiles(1).name));
    imgSize = size(sampleImg);
    numPixels = imgSize(1) * imgSize(2);

    X = zeros(numPixels, numImages);
    Y = zeros(1, numImages);

    for i = 1:numImages
        imgPath = fullfile(imgFiles(i).folder, imgFiles(i).name);
        img = imread(imgPath);

        if size(img, 3) == 3
            img = rgb2gray(img);
        end

        X(:, i) = double(img(:)) / 255;

        fileName = imgFiles(i).name;
        tokens = regexp(fileName, '\.(\w{2})\d', 'tokens');
        labelStr = tokens{1}{1};
        labelIdx = find(strcmp(labelNames, labelStr));
        Y(i) = labelIdx;
    end
    % Original image dimensions
    originalHeight = 256;
    originalWidth = 256;

% Target image dimensions
        newHeight = 50;
        newWidth = 50;

% Initialize the output matrix
X_re = zeros(newHeight * newWidth, size(X, 2)); % 2500 x 213

% Loop through each column (image) in the input matrix
for i = 1:size(X, 2)
    % Extract the i-th column and reshape it into a 256x256 image
    originalImage = reshape(X(:, i), [originalHeight, originalWidth]);
    
    % Resize the image to 26x26
    resizedImage = imresize(originalImage, [newHeight, newWidth]);
    
    % Flatten the resized image into a vector and store it in the output matrix
    X_re(:, i) = resizedImage(:);
end
end
