addpath('./functions');
[X_train, Y_train, X_test, Y_test] = loadMNIST('./MNIST/raw/');
% Example data
n = 10000; % Number of samples
c = 10;   % Number of clusters

% Random cluster assignment matrix (one-hot encoding)
Y = zeros(n, c);
for i = 1:n
    Y(i, randi(c)) = 1; % Randomly assign each sample to a cluster
end


% Calculate accuracy
acc = calculate_accuracy(Y_test, Y);
fprintf('Clustering Accuracy: %.2f%%\n', acc * 100);