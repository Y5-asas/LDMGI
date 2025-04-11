addpath('./functions');

% Load USPS data
%[X_train, Y_train, X_test, Y_test] = loadUSPS('./USPS/usps.h5');
% X_comb = [X_train, X_test];  % 合并数据
% Y_comb = [Y_train, Y_test];  % 合并标签
% Load USPS data


% Load JAFFE data 
load('JAFFE.mat');
X_comb =  X_JAFFE;
Y_comb = Y_JAFFE;
Y_comb = Y_comb-1;
% Load JAFFE data

% Load MNIST-T data
%dataPath = './MNIST/raw/';

% Load MNIST-T data
%[X_train, Y_train, X_test, Y_test] = loadMNIST(dataPath);


%X_train_flattened = reshape(X_train, 28*28, 60000); % X_train: (28*28, 60000)
%X_test_flattened = reshape(X_test, 28*28, 10000); % X_test: (28*28, 10000)

%Y_comb = Y_test(1:5000);
%X_comb = X_test_flattened(:, 1:5000); % Coresponds with the papaer MNIST-T
% Load MNIST-T data


c = 10;          % 聚类数量
k = 5;           % 局部邻域大小
lambda_list = [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8];  % 待测试的 lambda 值
output_dir = './Experiment_data/JAFFE';  % 结果保存目录
mkdir(output_dir);  % 创建文件夹（如果不存在）

% 循环测试每个 lambda
for lambda = lambda_list
    fprintf('Running experiment with lambda = %.0e...\n', lambda);
    
    % 聚类
    Y = ldmgi_clustering(X_comb, c, 'k', k, 'lambda_input', lambda);
    
    % 评估
    cost_matrix = calculate_cost_matrix(Y, Y_comb,c);
    [assignment, ~] = munkres(-cost_matrix);
    [acc, ~, ~] = calculate_accuracy_with_assignment(Y, Y_comb, assignment);
    nmi = calculate_NMI(Y_comb, Y);
    
    % 打印结果
    fprintf('Lambda: %.0e, ACC: %.4f, NMI: %.4f\n', lambda, acc, nmi);
    
    % 保存 Y 到 .mat 文件
    lambda_str = strrep(sprintf('%.0e', lambda), '+', '');  % 格式化 lambda 字符串（例如 1e4）
    filename = sprintf('Y_JAFFE_lambda_%s.mat', lambda_str);
    save(fullfile(output_dir, filename), 'Y');
end
