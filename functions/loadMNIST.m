function [X_train, Y_train, X_test, Y_test] = loadMNIST(dataPath)
    % Usage of this function, just load the MNIST datapath like MNIST/raw
    % dataFolder = 'MNIST/raw';
    % [X_train, Y_train, X_test, Y_test] = loadMNIST(dataFolder);
    trainImagesFile = fullfile(dataPath, 'train-images-idx3-ubyte');
    trainLabelsFile = fullfile(dataPath, 'train-labels-idx1-ubyte');
    testImagesFile = fullfile(dataPath, 't10k-images-idx3-ubyte');
    testLabelsFile = fullfile(dataPath, 't10k-labels-idx1-ubyte');

    % 加载训练集图像
    fid = fopen(trainImagesFile, 'r');
    fread(fid, 16, 'uint8'); % 跳过头部信息
    X_train = reshape(fread(fid, [28 * 28, 60000], 'uint8'), [28, 28, 60000]);
    fclose(fid);

    % 加载训练集标签
    fid = fopen(trainLabelsFile, 'r');
    fread(fid, 8, 'uint8'); % 跳过头部信息
    Y_train = fread(fid, [1, 60000], 'uint8');
    fclose(fid);

    % 加载测试集图像
    fid = fopen(testImagesFile, 'r');
    fread(fid, 16, 'uint8'); % 跳过头部信息
    X_test = reshape(fread(fid, [28 * 28, 10000], 'uint8'), [28, 28, 10000]);
    fclose(fid);

    % 加载测试集标签
    fid = fopen(testLabelsFile, 'r');
    fread(fid, 8, 'uint8'); % 跳过头部信息
    Y_test = fread(fid, [1, 10000], 'uint8');
    fclose(fid);
end