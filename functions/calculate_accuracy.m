function acc = calculate_accuracy(true_labels, pred_labels)
    % 输入:
    %   true_labels: 真实标签，n*c
    %   pred_labels: 聚类输出标签，1×n 或 n×1
    % 输出:
    %   acc: 聚类准确率 (Clustering Accuracy)
    [~, pred_labels] = max(pred_labels, [], 2);
    pred_labels = pred_labels-1;
    true_labels = true_labels(:);
    pred_labels = pred_labels(:);
    
    % 获取所有不同标签
    labels_true = unique(true_labels);
    labels_pred = unique(pred_labels);
    
    % 统一标签为 1 到 n
    true_mapped = zeros(size(true_labels));
    pred_mapped = zeros(size(pred_labels));
    
    for i = 1:length(labels_true)
        true_mapped(true_labels == labels_true(i)) = i;
    end
    for i = 1:length(labels_pred)
        pred_mapped(pred_labels == labels_pred(i)) = i;
    end
    
    % 构建混淆矩阵
    n = length(true_labels);
    D = max(max(true_mapped), max(pred_mapped));
    cost_matrix = zeros(D, D);
    for i = 1:n
        cost_matrix(pred_mapped(i), true_mapped(i)) = cost_matrix(pred_mapped(i), true_mapped(i)) + 1;
    end

    % 使用匈牙利算法找最优标签映射 (matchpairs 需要 Optimization Toolbox)
    cost_matrix = max(cost_matrix(:)) - cost_matrix;  % 转为最小化问题
    
    [assignment, ~] = matchpairs(cost_matrix, 0);  % assignment 是 c×2 的映射

    % 重新映射预测标签
    new_pred = zeros(n,1);
    for i = 1:size(assignment,1)
        old_label = assignment(i,1);
        new_label = assignment(i,2);
        new_pred(pred_mapped == old_label) = new_label;
    end
    
    % 计算准确率
    acc = sum(new_pred == true_mapped) / n;
end