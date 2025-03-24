function nmi = calculate_NMI(true_labels, pred_labels)
    true_labels = true_labels(:);
    pred_labels = pred_labels(:);
    n = length(true_labels);

    % 获取所有标签（转为 1~c 的编号）
    label_true = unique(true_labels);
    label_pred = unique(pred_labels);
    c_true = length(label_true);
    c_pred = length(label_pred);

    % 构造混淆矩阵（交集）
    conf_mat = zeros(c_pred, c_true);
    for i = 1:c_pred
        for j = 1:c_true
            conf_mat(i,j) = sum(pred_labels == label_pred(i) & true_labels == label_true(j));
        end
    end

    % 每类的样本数
    t_l = sum(conf_mat, 2);     % 预测标签每类样本数
    t_hat_h = sum(conf_mat, 1); % 真实标签每类样本数

    % 互信息 MI
    MI = 0;
    for i = 1:c_pred
        for j = 1:c_true
            if conf_mat(i,j) > 0
                MI = MI + conf_mat(i,j) * log((conf_mat(i,j) * n) / (t_l(i) * t_hat_h(j)));
            end
        end
    end

    % 计算 H(pred) 和 H(true)
    H_pred = sum(t_l .* log(t_l / n + eps));         % 加 eps 避免 log(0)
    H_true = sum(t_hat_h .* log(t_hat_h / n + eps)); % 加 eps 避免 log(0)

    % 最终 NMI
    nmi = MI / sqrt(H_pred * H_true);
end