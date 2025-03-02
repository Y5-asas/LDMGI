function [Y, local_cliques] = initial_clique(n, c, X, k)
    % INPUTS:
    %   n: Number of samples
    %   c: Number of clusters
    %   X: Data matrix of size (m x n), where m is the number of features
    %      and n is the number of samples
    %   k: Number of neighbors for local cliques (including the point itself)
    %
    % OUTPUTS:
    %   Y: Cluster assignment matrix of size (n x c)
    %   local_cliques: Cell array of size (n x 1), where each cell contains
    %                  the indices of the k-nearest neighbors for each point

    % Step 1: Randomly assign each sample to a cluster
    cluster_assignments = randi([1, c], n, 1);  % Random integers between 1 and c

    % Step 2: Initialize the Y matrX_iix
    Y = zeros(n, c);  % Initialize Y as an n x c matrix of zeros
    for i = 1:n
        Y(i, cluster_assignments(i)) = 1;  % Set the assigned cluster to 1
    end

    % Step 3: Find k-nearest neighbors for each point
    % Transpose X to make it n x m (samples as rows, features as columns)
    X_transposed = X';  % Transpose X to n x m
    [idx, ~] = knnsearch(X_transposed, X_transposed, 'K', k);  % Find k-nearest neighbors

    % Step 4: Construct local cliques
    local_cliques = cell(n, 1);  % Store local cliques in a cell array
    for i = 1:n
        local_cliques{i} = idx(i, :);  % Store the indices of the k-nearest neighbors
    end
end