function Y = ncut_clustering(X, c, k, sigma)
% NCUT_CLUSTERING Spectral clustering using Normalized Cuts
%   X     : d × n feature matrix
%   c     : number of clusters
%   k     : number of neighbors (k-NN)
%   sigma : Gaussian kernel bandwidth

    [d, n] = size(X);

    % Step 1: Construct A using k-nearest neighbors
    A = zeros(n, n);
    dist_mat = pdist2(X', X');  % n × n

    for i = 1:n
        [~, idx] = sort(dist_mat(i, :), 'ascend');
        neighbors = idx(2:k+1);  % exclude itself
        for j = neighbors
            A(i, j) = exp(-norm(X(:,i) - X(:,j))^2 / (sigma^2));
            A(j, i) = A(i, j);  % symmetric
        end
    end

    % Step 2: Construct D and normalized Laplacian
    D = diag(sum(A, 2));
    D_inv_sqrt = diag(1 ./ sqrt(diag(D) + eps));  % add eps to avoid /0
    L = eye(n) - D_inv_sqrt * A * D_inv_sqrt;

    % Step 3: Eigen decomposition (L is symmetric, use eig)
    [V, D_eig] = eig(L);
    [~, idx] = sort(diag(D_eig), 'ascend');
    G_star = V(:, idx(1:c));  % n × c (use smallest eigenvalues)

    % Step 4: Run K-means
    idx = kmeans(G_star, c, 'Replicates', 10);
    Y = full(sparse(1:n, idx, 1, n, c));  % One-hot
end
