function L_i = compute_L_i(X, local_cliques, i, lambda)
    % INPUTS:
    %   X: Global data matrix of size (m x n), where m is the number of features
    %   local_cliques: Cell array of size (n x 1), where each cell contains
    %                  the indices of the k-nearest neighbors for each point
    %   i: Index of the clique for which to compute L_i
    %   lambda: Regularization parameter
    %
    % OUTPUT:
    %   L_i: Local Laplacian matrix for the i-th clique

    % Step 1: Extract the local data matrix X_i
    clique_indices = local_cliques{i};  % Indices of the k-nearest neighbors

    X_i = X(:, clique_indices);        % Local data matrix for the clique X_i size(m,k)
    


    % Step 2: Center the local data matrix X_i
                  
    k = size(X_i, 2);             % Number of points in the clique     
    H_k = eye(k) - (1/k) * ones(k, k); % Centering matrix
    X_i_centered =  X_i*H_k;          % Centered local data matrix

    % Debug: Check the size of X_i_centered
    fprintf('Size of X_i_centered: %d x %d\n', size(X_i_centered, 1), size(X_i_centered, 2));

    % Step 3: Compute L_i
    % Ensure the regularization term has the correct size
    regularization_term = lambda * eye(k);  % Size should be k x k
    L_i = H_k * ((X_i_centered' * X_i_centered + regularization_term) \ H_k);
    
    % Debug: Check the size of L_i
    fprintf('Size of L_i: %d x %d\n', size(L_i, 1), size(L_i, 2));
end
