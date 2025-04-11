function [X_train, Y_train, X_test, Y_test] = loadUSPS(dataPath)
    X_train = h5read(dataPath, '/train/data');
    Y_train = h5read(dataPath, '/train/target');
    X_test  = h5read(dataPath, '/test/data');
    Y_test  = h5read(dataPath, '/test/target');

    X_train = double(X_train);
    X_test  = double(X_test);

    Y_train = double(Y_train(:));
    Y_test  = double(Y_test(:));

    Y_train = reshape(Y_train, 1, []);
    Y_test  = reshape(Y_test, 1, []);

    X_train = X_train / 255;
    X_test  = X_test / 255;
end
