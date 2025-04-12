clc; clear;
addpath('./functions');

% === 加载 JAFFE 数据 ===
load('JAFFE.mat');  % X_JAFFE: 676 × 213, Y_JAFFE: 1 × 213

X = X_JAFFE;
Y_Label = Y_JAFFE;
n = size(X, 2);
c = 10;  % JAFFE 有 10 个人

% === 定义 gamma 参数集合（与 LDMGI 中 lambda 一致）
gamma_list = [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8];

% === 创建输出文件夹
output_folder = './JAFFE_DiskMeans_Result/';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% === 初始化结果记录表
results = [];


Y_JAFFE_test = Y_Label-1;

for gamma = gamma_list
    fprintf('\nRunning DiskMeans for gamma = %g\n', gamma);
    
    % === 调用 DiskMeans 聚类函数 ===
    Y_disk = diskmeans_clustering(X, c, gamma);  % 自定义函数见下方

    % === 聚类评估 ===
    cost_matrix = calculate_cost_matrix(Y_disk, Y_JAFFE_test, c);
    [assignment, ~] = munkres(-cost_matrix);
    [acc, ~, ~] = calculate_accuracy_with_assignment(Y_disk, Y_JAFFE_test, assignment);
    nmi = calculate_NMI(Y_JAFFE_test, Y_disk);

    fprintf('[DiskMeans] Gamma = %g | ACC: %.4f | NMI: %.4f\n', gamma, acc, nmi);

    % === 保存结果矩阵 ===
    gamma_str = num2str(gamma, '%.0e');
    filename = ['Y_JAFFE_DiskMeans_Gamma_' gamma_str '.mat'];
    save(fullfile(output_folder, filename), 'Y_disk');

    % === 记录结果 ===
    results = [results; table(gamma, acc, nmi)];
end

% === 保存汇总表 ===
result_file = fullfile(output_folder, 'JAFFE_DiskMeans_Result.xlsx');
writetable(results, result_file);
fprintf('\n✅ 所有结果保存至 %s\n', result_file);
