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

% Calculate the cost matrix
cost_matrix = calculate_cost_matrix(Y, Y_test,10);
[assignment, cost] = munkres(-cost_matrix);

[acc,predicted_labels,remapped_Y] = calculate_accuracy_with_assignment(Y,Y_test,assignment);
