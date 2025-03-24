addpath('./functions');
[X_train, Y_train, X_test, Y_test] = loadMNIST('/MATLAB Drive/MNIST/MNIST/raw');

% Example data
n = 10000; % Number of samples
c = 10;    % Number of clusters

% Random cluster assignment matrix (one-hot encoding)
Y = zeros(n, c);
for i = 1:n
    Y(i, randi(c)) = 1; % Randomly assign each sample to a cluster
end

% Calculate Normalized Mutual Information (NMI)
nmi = calculate_NMI(Y_test, Y);  
fprintf('Normalized Mutual Information (NMI): %.4f\n', nmi);
