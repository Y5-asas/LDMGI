function cost_matrix = calculate_cost_matrix(Y, Y_test, num_classes)
    % Inputs:
    %   Y: n x c matrix (one-hot encoding of predicted labels)
    %   Y_test: 1 x n vector (ground truth labels, 0-based indexing)
    %   num_classes: Number of classes (c)
    %
    % Output:
    %   cost_matrix: c x c matrix where cost_matrix(i,j) represents the cost
    %                of assigning predicted class i to true class j

    % Initialize the cost matrix
    cost_matrix = zeros(num_classes, num_classes);

    % Convert one-hot encoding to predicted class labels
    [~, predicted_labels] = max(Y, [], 2); % Find the column index of the maximum value in each row
    predicted_labels = predicted_labels - 1; % Convert to 0-based indexing
    
    % Compute the cost matrix
    for i = 1:length(predicted_labels)
        pred_class = predicted_labels(i); % Predicted class for this data point
        true_class = Y_test(i);           % True class for this data point
        cost_matrix(pred_class + 1, true_class + 1) = cost_matrix(pred_class + 1, true_class + 1) + 1;
    end

    % Normalize the cost matrix (optional)
    % This ensures that the costs are proportional to the frequency of misclassifications
    cost_matrix = cost_matrix / sum(cost_matrix(:));
end