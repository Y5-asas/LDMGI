function Y = ldmgi_clustering(X, c, varargin)
% LDMGI_CLUSTERING 基于局部判别模型和全局集成的聚类算法
%
% 输入:
%   X: m x n 数据矩阵 (m: 特征维度, n: 样本数)
%   c: 聚类数量
%   varargin: 可选参数
%     - 'k': 局部邻域大小 (默认: 5)
%     - 'lambda': 正则化参数 (默认: 1e8)
%     - 'max_iters': 最大迭代次数 (默认: 500)
%     - 'epsilon': 收敛阈值 (默认: 1e-2)
%
% 输出:
%   Y: n x c 的 one-hot 编码聚类标签矩阵

% 解析可选参数
params = inputParser;
addParameter(params, 'k', 5, @isnumeric);
addParameter(params, 'lambda_input', 1e8, @isnumeric);
addParameter(params, 'max_iters', 500, @isnumeric);
addParameter(params, 'epsilon', 1e-2, @isnumeric);
parse(params, varargin{:});

k = params.Results.k;
lambda_input = params.Results.lambda_input;
max_iters = params.Results.max_iters;
epsilon = params.Results.epsilon;

n = size(X, 2);  % 样本数

%% Step 1: 初始化局部邻域和全局拉普拉斯矩阵 L
[~, local_cliques] = initial_clique(n, c, X, k);  % 初始化局部邻域
L_total = zeros(n, n);  % 全局 L 初始化为零矩阵

for i = 1:n
    % 计算局部拉普拉斯矩阵 L_i
    L_i = compute_L_i(X, local_cliques, i, lambda_input);
    
    % 构造选择矩阵 S_i
    clique_indices = local_cliques{i};
    S_i = zeros(n, length(clique_indices));
    for j = 1:length(clique_indices)
        S_i(clique_indices(j), j) = 1;
    end
    
    % 更新全局 L
    L_total = L_total + S_i * L_i * S_i';
end
disp('Computed Global Laplacian Matrix L:');
disp(size(L_total));  % Debug: 确保 L 的尺寸正确 (n × n)
%% Step 2: 特征分解和谱旋转
[U, Lambda] = eig(L_total);
eigenvalues = diag(Lambda);
U = real(U); % 取本征向量的实部

[~, sorted_indices] = sort(eigenvalues, 'ascend');
G_star = U(:, sorted_indices(2:c+1));  % 去掉最小特征值，取前 c 个(因为直接跳过第一个 所以直接从U里面取2到c+1)

% 计算 Y* (公式 23)
D_inv_sqrt = diag(1 ./ sqrt(diag(G_star * G_star')));
Y_star = D_inv_sqrt * G_star;

%% Step 3: 迭代优化 (谱旋转 + K-means)
Y = zeros(n, c);
prev_loss = inf;

for iter = 1:max_iters
    % 计算旋转矩阵 R
    [Uq, ~, Vq] = svd(Y_star' * Y);
    R = Vq * Uq';
    
    % 更新 Y (K-means)
    Y_new = Y_star * R;
    idx = kmeans(Y_new, c, 'Replicates', 10);
    Y_new = full(sparse(1:n, idx, 1, n, c));  % 快速 one-hot 编码
    
    % 检查收敛
    loss = norm(Y_new - Y_star * R, 'fro')^2;
    if abs(prev_loss - loss) < epsilon
        break;
    end
    prev_loss = loss;
    Y = Y_new;
end
end