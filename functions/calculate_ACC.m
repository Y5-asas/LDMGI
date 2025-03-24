function acc = calculate_ACC(true_labels, pred_labels)
    % 输入:
    %   true_labels: 真实标签，n×1 或 one-hot 编码矩阵 (n×c)
    %   pred_labels: 聚类结果标签，n×1 或 one-hot 编码矩阵 (n×c)
    % 输出:
    %   acc: 聚类准确率 (Clustering Accuracy)

    % --- 步骤1: 确保输入为类别标签向量 ---
    % 处理 one-hot 编码的预测标签
    if size(pred_labels, 2) > 1
        [~, pred_labels] = max(pred_labels, [], 2);
        pred_labels = pred_labels - 1; % 转换为 0-based 标签
    end
    % 确保标签为列向量
    true_labels = true_labels(:);
    pred_labels = pred_labels(:);

    % --- 步骤2: 统一标签为连续整数 (1~C) ---
    % 真实标签映射 (p_i)
    [unique_true, ~, true_mapped] = unique(true_labels);
    true_mapped = true_mapped - 1; % 转换为 0-based (公式中的 p_i)
    
    % 预测标签映射 (q_i)
    [unique_pred, ~, pred_mapped] = unique(pred_labels);
    pred_mapped = pred_mapped - 1; % 转换为 0-based (公式中的 q_i)
    
    % --- 步骤3: 构建混淆矩阵 (共现次数) ---
    C = max(true_mapped) + 1; % 真实类别数
    K = max(pred_mapped) + 1; % 预测类别数
    confusion_matrix = zeros(K, C);
    for i = 1:length(true_labels)
        confusion_matrix(pred_mapped(i)+1, true_mapped(i)+1) = ...
            confusion_matrix(pred_mapped(i)+1, true_mapped(i)+1) + 1;
    end

    % --- 步骤4: 匈牙利算法找到最优映射 (map(q_i)) ---
    cost_matrix = max(confusion_matrix(:)) - confusion_matrix; % 转为最小化问题
    [assignment, ~] = matchpairs(cost_matrix, 0); % 使用 Optimization Toolbox

    % --- 步骤5: 重新映射预测标签 ---
    new_pred = zeros(size(pred_mapped));
    for i = 1:size(assignment, 1)
        old_label = assignment(i, 1) - 1; % 原预测标签 (0-based)
        new_label = assignment(i, 2) - 1; % 映射后的标签 (0-based)
        new_pred(pred_mapped == old_label) = new_label;
    end

    % --- 步骤6: 计算ACC ---
    acc = sum(new_pred == true_mapped) / length(true_labels);
end