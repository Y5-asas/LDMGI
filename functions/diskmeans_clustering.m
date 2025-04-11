function Y = diskmeans_clustering(X, c, gamma)
% DISKMEANS_CLUSTERING: Discriminative K-means clustering
%
% Input:
%   X     - d × n data matrix
%   c     - number of clusters
%   gamma - regularization parameter
%
% Output:
%   Y     - n × c one-hot encoded clustering matrix

    [d, n] = size(X);
    
    % Step 1: compute matrix M
    XtX = X' * X;  % n × n
    M = eye(n) - inv(eye(n) + (1/gamma) * XtX);  % n × n

    % Step 2: eigen-decomposition
    [V, D] = eig(M);
    [~, idx] = sort(diag(D), 'descend');
    V = V(:, idx(1:c));  % take top-c eigenvectors

    % Step 3: run K-means on low-dim embedding
    Y_star = V;  % n × c
    idx = kmeans(Y_star, c, 'Replicates', 10);

    % Step 4: convert to one-hot matrix
    Y = full(sparse(1:n, idx, 1, n, c));
end
