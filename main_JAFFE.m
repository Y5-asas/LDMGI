addpath('./functions');

% === 加载 JAFFE 数据 ===
load('JAFFE.mat');  % 包含 X_JAFFE, Y_JAFFE

X = X_JAFFE;
Y_Label = Y_JAFFE;
n = size(X, 2);
c = 10;     % JAFFE 有 10 个人
k = 5;      % 邻居数

% === 定义 lambda 参数集合 ===
lambda_list = [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8];

% === 创建结果保存目录 ===
output_folder = './JAFFE_Result/';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% === 初始化评估结果表 ===
results = [];

% === 遍历每个 lambda 并执行聚类 ===
for lambda = lambda_list
    fprintf('\nRunning LDMGI for JAFFE: lambda = %g\n', lambda);
    
    % --- 调用你的封装函数 ---
    Y = ldmgi_clustering(X, c, ...
        'k', k, ...
        'lambda_input', lambda, ...
        'max_iters', 500, ...
        'epsilon', 1e-2);

    % --- 聚类后评估 ---
    Y_JAFFE_test = Y_Label-1;
    cost_matrix = calculate_cost_matrix(Y, Y_JAFFE_test, c);
    [assignment, ~] = munkres(-cost_matrix);
    [acc, ~, ~] = calculate_accuracy_with_assignment(Y, Y_JAFFE_test, assignment);
    nmi = calculate_NMI(Y_Label, Y);

    fprintf('Lambda = %g | ACC = %.4f | NMI = %.4f\n', lambda, acc, nmi);

    % --- 保存 Y ---
    lambda_str = num2str(lambda, '%.0e');  % e.g. 1e-08
    filename = ['Y_JAFFE_Lambda_' lambda_str '.mat'];
    save(fullfile(output_folder, filename), 'Y');

    % --- 记录结果 ---
    results = [results; table(lambda, acc, nmi)];
end

% === 保存最终评估表格 ===
result_file = fullfile(output_folder, 'JAFFE_Result.xlsx');
writetable(results, result_file);
fprintf('\n✅ 所有结果保存至 %s\n', result_file);
