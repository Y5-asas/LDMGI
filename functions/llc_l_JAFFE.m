load('JAFFE.mat');
X = X_JAFFE;
Y_Label = Y_JAFFE - 1;
c = 10;
k = 5;
eta = 1e-3;

Y_llcl = llc_linear_clustering(X, c, k, eta);

% 评估
cost_matrix = calculate_cost_matrix(Y_llcl, Y_Label, c);
[assignment, ~] = munkres(-cost_matrix);
[acc_llcl, ~, ~] = calculate_accuracy_with_assignment(Y_llcl, Y_Label, assignment);
nmi_llcl = calculate_NMI(Y_Label, Y_llcl);

fprintf('[LLC-L] ACC: %.4f | NMI: %.4f\n', acc_llcl, nmi_llcl);
