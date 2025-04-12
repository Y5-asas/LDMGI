addpath('./functions');

% === 加载 JAFFE 数据 ===
load('JAFFE.mat');
X = X_JAFFE;
Y_Label = Y_JAFFE - 1;  % 转成 0-based label
n = size(X, 2);
c = 10;
k = 5;         % 固定邻居数
eta = 1e-3;    % 固定正则化参数

% === sigma 参数列表 ===
sigma_list = [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8];

% === 输出文件夹
output_folder = './JAFFE_LLCG_Result/';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% === 初始化结果表格
results = [];
Y_JAFFE_test = Y_Label - 1;

for sigma = sigma_list
    fprintf('\n[LLC-G] Running sigma = %g\n', sigma);

    % 聚类执行
    Y_llcg = llc_gaussian_clustering(X, c, k, eta, sigma);

    % 聚类评估
    cost_matrix = calculate_cost_matrix(Y_llcg, Y_Label, c);
    [assignment, ~] = munkres(-cost_matrix);
    [acc, ~, ~] = calculate_accuracy_with_assignment(Y_llcg, Y_Label, assignment);
    nmi = calculate_NMI(Y_Label, Y_llcg);

    fprintf('[LLC-G] Sigma = %g | ACC: %.4f | NMI: %.4f\n', sigma, acc, nmi);

    % 保存结果
    sigma_str = num2str(sigma, '%.0e');
    mat_filename = ['Y_JAFFE_LLCG_Sigma_' sigma_str '.mat'];
    save(fullfile(output_folder, mat_filename), 'Y_llcg');

    % 保存结果记录
    results = [results; table(sigma, acc, nmi)];
end

% === 保存最终表格
result_file = fullfile(output_folder, 'JAFFE_LLCG_Result.xlsx');
writetable(results, result_file);
fprintf('\n 所有 LLC-G 结果已保存至 %s\n', result_file);
