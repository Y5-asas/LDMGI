function Y = llc_gaussian_clustering(X, c, k, eta, sigma)
% LLC-G: Local Learning Clustering using Gaussian kernel
% Inputs:
%   X     : d × n feature matrix
%   c     : number of clusters
%   k     : number of neighbors
%   eta   : regularization parameter
%   sigma : Gaussian kernel bandwidth
% Output:
%   Y     : n × c one-hot clustering result

[d, n] = size(X);
A = zeros(n, n);  % affinity matrix

% === Step 1: Construct affinity matrix A ===
for i = 1:n
    xi = X(:, i);  % 当前点
    dist = vecnorm(X - xi).^2;  % 欧式距离平方
    dist(i) = inf;  % 排除自身
    [~, nn_idx] = mink(dist, k);  % 取前 k 个邻居索引
    
    % 取邻居子集
    X_nn = X(:, nn_idx);  % d × k
    k_i = exp(-pdist2(xi', X_nn').^2 / (sigma^2));  % 1 × k 高斯核向量
    Ki = exp(-pdist2(X_nn', X_nn').^2 / (sigma^2)); % k × k 子核矩阵

    % 计算 α_i = k_i^T (Ki + ηI)^-1
    alpha_i = (k_i / (Ki + eta * eye(k)))';  % k × 1

    for j = 1:k
        A(i, nn_idx(j)) = alpha_i(j);
    end
end

% === Step 2: 构造拉普拉斯矩阵并谱聚类 ===
L = (eye(n) - A)' * (eye(n) - A);  % LLC 中定义的 L

% === Step 3: 特征分解 + K-means ===
[U, D] = eig(L);
[~, idx] = sort(diag(D), 'ascend');
G = U(:, idx(1:c));  % 取前 c 个特征向量

idx = kmeans(G, c, 'Replicates', 10);
Y = full(sparse(1:n, idx, 1, n, c));
end
