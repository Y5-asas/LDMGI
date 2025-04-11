function Y = llc_linear_clustering(X, c, k, eta)
% LLC-L: Local Learning Clustering using Linear kernel
% Inputs:
%   X   : d × n feature matrix
%   c   : number of clusters
%   k   : number of neighbors
%   eta : regularization parameter
% Output:
%   Y   : n × c one-hot clustering result

[d, n] = size(X);
A = zeros(n, n);  % affinity matrix

% === Step 1: Build local kernel regression model for each point ===
for i = 1:n
    xi = X(:, i);
    dist = vecnorm(X - xi).^2;
    dist(i) = inf;
    [~, nn_idx] = mink(dist, k);  % k-NN excluding xi

    % Linear kernel
    ki = X(:, nn_idx)' * xi;          % k × 1
    Ki = X(:, nn_idx)' * X(:, nn_idx); % k × k

    % α_i = kiᵗ (Ki + ηI)^(-1)
    alpha_i = (ki' / (Ki + eta * eye(k)))';  % k × 1

    % Fill in affinity matrix A
    for j = 1:k
        A(i, nn_idx(j)) = alpha_i(j);
    end
end

% === Step 2: Laplacian + spectral embedding ===
L = (eye(n) - A)' * (eye(n) - A);  % LLC 的 L

% === Step 3: eigen-decomposition + k-means ===
[U, D] = eig(L);
[~, idx] = sort(diag(D), 'ascend');
G = U(:, idx(1:c));

idx = kmeans(G, c, 'Replicates', 10);
Y = full(sparse(1:n, idx, 1, n, c));
end
