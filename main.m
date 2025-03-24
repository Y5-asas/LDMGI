addpath('./functions');
dataPath = './MNIST/raw/';
% 加载 MNIST 数据
[X_train, Y_train, X_test, Y_test] = loadMNIST(dataPath);


X_train_flattened = reshape(X_train, 28*28, 60000); % X_train: (28*28, 60000)
X_test_flattened = reshape(X_test, 28*28, 10000); % X_test: (28*28, 10000)
X_comb = [X_train_flattened,X_test_flattened]; % X_comb: (28*28, 70000)
Y_comb = [Y_train ,Y_test]; % Y_comb: (1, 70000). Here the Y_comb with number from 0-9 
Y_MNIST_T_test = Y_test(1:5000);

X_test_T = X_test_flattened(:, 1:5000); % Coresponds with the papaer MNIST-T
n = size(X_test_T,2);  % Number of samples
c = 10;    % Number of clusters
X = X_test_T;  % X_comb with consideration of use MNIST_S or MNIST_T instead 
k = 5;    % Number of neighbors for local cliques
%%
% Call the function
[Y, local_cliques] = initial_clique(n, c, X, k);

Y_init = Y;
G = G_for_Y(Y);

%% Step 1: 初始化全局 L 并计算所有 L_i
lambda = 1e-6;  % Regularization parameter
L_total = zeros(n, n);  % 全局 L 初始化为零矩阵

for i = 1:n
    % 计算局部拉普拉斯矩阵 L_i
    L_i = compute_L_i(X, local_cliques, i, lambda);
    
    % 获取 S_i（选择矩阵）
    clique_indices = local_cliques{i};  % 该 clique 的样本索引
    S_i = zeros(n, length(clique_indices));
    
    for j = 1:length(clique_indices)
        S_i(clique_indices(j), j) = 1;  % 选择矩阵 S_i
    end
    
    % 计算 L_i 在全局 L 的投影
    L_total = L_total + S_i * L_i * S_i';
end


disp('Computed Global Laplacian Matrix L:');
disp(size(L_total));  % Debug: 确保 L 的尺寸正确 (n × n)


[U, Lambda] = eig(L_total);
U_1 = U;

% 获取特征值
eigenvalues = diag(Lambda);  % 提取特征值
[sorted_eigenvalues, sorted_indices] = sort(eigenvalues, 'ascend');  % 升序排列

% 去掉最小特征值（对应于平凡解）
U = U(:, sorted_indices(2:end));  
Lambda = Lambda(sorted_indices(2:end), sorted_indices(2:end));  


G_star = U(:, 1:c);  % 取前 c 个特征向量

% 计算 Y* (公式 23)
D_inv_sqrt = diag(1 ./ sqrt(diag(G_star * G_star')));
Y_star = D_inv_sqrt * G_star;  

% 运行 K-means
num_clusters = c;
idx = kmeans(Y_star, num_clusters, 'Replicates', 10);


% 转换为 one-hot 编码
Y = zeros(size(G_star, 1), num_clusters);
for i = 1:length(idx)
    Y(i, idx(i)) = 1;
end
% disp(Y);

max_iters = 500;  % 设置最大迭代次数
epsilon = 1e-4;  % 设定收敛阈值
prev_loss = inf; % 记录上一次损失值
iter_number = 0;
for iter = 1:max_iters
    iter_number = iter_number+ 1 ;
    % 计算旋转矩阵 R
    M = Y_star' * Y;
    [Uq, ~, Vq] = svd(M);
    R = Vq * Uq';

    % 更新 Y
    Y_new = Y_star * R;
    idx = kmeans(Y_new, num_clusters, 'Replicates', 10);
    Y_new = zeros(size(G_star, 1), num_clusters);
    for i = 1:length(idx)
        Y_new(i, idx(i)) = 1;
    end

    % 计算收敛误差 ||Y - Y^* R||^2
    loss = norm(Y_new - Y_star * R, 'fro')^2;
    if abs(prev_loss - loss) < epsilon
        break;  % 如果损失收敛，则停止迭代
    end
    prev_loss = loss;
    Y = Y_new;
end

I_approx = R' * R;

cost_matrix = calculate_cost_matrix(Y, Y_MNIST_T_test,c);
[assignment, cost] = munkres(-cost_matrix);

[acc,predicted_labels,remapped_Y] = calculate_accuracy_with_assignment(Y, Y_MNIST_T_test,assignment);
nmi = calculate_NMI(Y_MNIST_T_test, Y);  

fprintf('Normalized Mutual Information (NMI): %.4f\n', nmi);
