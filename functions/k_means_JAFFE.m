addpath('./functions');


% === 加载 JAFFE 数据 ===
load('JAFFE.mat');  % 包含 X_JAFFE, Y_JAFFE

X = X_JAFFE;
Y_Label = Y_JAFFE;
n = size(X, 2);
c = 10;     % JAFFE 有 10 个人
k = 5;      % 邻居数


[idx, ~] = kmeans(X', c, 'Replicates', 10);
Y_kmeans = full(sparse(1:n, idx, 1, n, c));
Y_JAFFE_test = Y_Label-1;


cost_matrix = calculate_cost_matrix(Y_kmeans, Y_JAFFE_test, c);
[assignment, ~] = munkres(-cost_matrix);
[acc_kmeans, ~, ~] = calculate_accuracy_with_assignment(Y_kmeans, Y_JAFFE_test, assignment);
nmi_kmeans = calculate_NMI(Y_JAFFE_test, Y_kmeans);

fprintf('[K-means Baseline] ACC: %.4f | NMI: %.4f\n', acc_kmeans, nmi_kmeans);
