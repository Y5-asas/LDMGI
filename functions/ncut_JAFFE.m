addpath('./functions');

% === 加载 JAFFE 数据 ===
load('JAFFE.mat');  % X_JAFFE: 676 × 213, Y_JAFFE: 1 × 213
X = X_JAFFE;
Y_Label = Y_JAFFE;
n = size(X, 2);
c = 10;  % JAFFE 有 10 类
k = 5;   % 固定邻居数（可调）

% === σ 参数集合（对应 lambda sweep）
sigma_list = [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8];

% === 创建输出文件夹
output_folder = './JAFFE_Ncut_Result/';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% === 初始化记录表
results = [];

Y_JAFFE_test = Y_Label - 1;



for sigma = sigma_list
    fprintf('\nRunning Ncut for sigma = %g\n', sigma);
    
    % 聚类调用
    Y_ncut = ncut_clustering(X, c, k, sigma);

    % 聚类评估
    cost_matrix = calculate_cost_matrix(Y_ncut, Y_JAFFE_test, c);
    [assignment, ~] = munkres(-cost_matrix);
    [acc, ~, ~] = calculate_accuracy_with_assignment(Y_ncut, Y_JAFFE_test, assignment);
    nmi = calculate_NMI(Y_JAFFE_test, Y_ncut);

    fprintf('[Ncut] Sigma = %g | ACC = %.4f | NMI = %.4f\n', sigma, acc, nmi);

    % 保存聚类结果
    sigma_str = num2str(sigma, '%.0e');
    save_name = ['Y_JAFFE_Ncut_Sigma_' sigma_str '.mat'];
    save(fullfile(output_folder, save_name), 'Y_ncut');

    % 记录结果
    results = [results; table(sigma, acc, nmi)];
end

% 保存汇总表
result_file = fullfile(output_folder, 'JAFFE_Ncut_Result.xlsx');
writetable(results, result_file);
fprintf('\n✅ 所有结果保存至 %s\n', result_file);
