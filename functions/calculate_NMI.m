function nmi = calculate_NMI(true_labels, pred_labels)
    % 输入:
    %   true_labels: 真实标签，n×1 或 one-hot 编码矩阵 (n×c)
    %   pred_labels: 聚类结果标签，n×1 或 one-hot 编码矩阵 (n×c)
    % 输出:
    %   nmi: 标准化互信息 (Normalized Mutual Information)

    % --- 步骤1: 确保输入为类别标签向量 ---
    % 处理 one-hot 编码的预测标签
    if size(pred_labels, 2) > 1
        [~, pred_labels] = max(pred_labels, [], 2);
    end
    % 确保标签为列向量
    true_labels = true_labels(:);
    pred_labels = pred_labels(:);
    n = length(true_labels);

    % --- 步骤2: 构建联合分布矩阵 t_{l,h} ---
    % 获取唯一标签
    [unique_true, ~, true_indices] = unique(true_labels);
    [unique_pred, ~, pred_indices] = unique(pred_labels);
    c_true = length(unique_true); % 真实类别数
    c_pred = length(unique_pred); % 聚类类别数

    % 初始化联合分布矩阵
    joint_matrix = zeros(c_pred, c_true);
    for i = 1:n
        joint_matrix(pred_indices(i), true_indices(i)) = ...
            joint_matrix(pred_indices(i), true_indices(i)) + 1;
    end

    % --- 步骤3: 计算边缘分布 t_l 和 t̃_h ---
    t_l = sum(joint_matrix, 2); % 聚类簇样本数 (行和)
    t_h = sum(joint_matrix, 1); % 真实类别样本数 (列和)

    % --- 步骤4: 计算互信息 I(P,Q) ---
    I = 0;
    for l = 1:c_pred
        for h = 1:c_true
            if joint_matrix(l, h) > 0
                term = joint_matrix(l, h) * log((n * joint_matrix(l, h)) / (t_l(l) * t_h(h)));
                I = I + term;
            end
        end
    end
    I = I / n; % 标准化

    % --- 步骤5: 计算熵 H(P) 和 H(Q) ---
    H_P = -sum(t_l .* log(t_l / n + eps)) / n; % 添加 eps 避免 log(0)
    H_Q = -sum(t_h .* log(t_h / n + eps)) / n;

    % --- 步骤6: 计算 NMI ---
    if H_P == 0 || H_Q == 0
        nmi = 0; % 避免除以0
    else
        nmi = I / sqrt(H_P * H_Q);
    end
end
