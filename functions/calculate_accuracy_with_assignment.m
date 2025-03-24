function [accuracy,predicted_labels,remapped_Y] = calculate_accuracy_with_assignment(Y, Y_test, assignment)
    % Inputs:
    %   Y: n x c matrix (one-hot encoding of predicted labels)
    %   Y_test: 1 x n vector (ground truth labels, 0-based indexing)
    %   assignment: 1 x c vector (mapping from predicted classes to true classes)
    %
    % Output:
    %   accuracy: Scalar value representing the classification accuracy

    % Step 1: Remap the predicted one-hot encoding using the assignment vector
% Step 1: Remap the predicted one-hot encoding using the assignment matrix
    [n, c] = size(Y); % Number of data points and classes
    remapped_Y = zeros(n, c); % Initialize remapped matrix

    for i = 1:c
    % Find the true class assigned to predicted class i
        [row, true_class] = find(assignment(i, :)); % Find the column index of the '1' in row i
        if ~isempty(true_class) % Ensure there is a valid mapping
        remapped_Y(:, true_class) = remapped_Y(:, true_class) + Y(:, i);
        end
    end
    
    % Step 2: Convert remapped one-hot encoding to predicted class labels
    [~, predicted_labels] = max(remapped_Y, [], 2); % Find the column index of the maximum value in each row
    predicted_labels = predicted_labels - 1; % Convert to 0-based indexing
    predicted_labels = predicted_labels'; % Transpose to match Y_test orientation

    % Step 3: Compare predicted labels with ground truth
    correct_predictions = sum(predicted_labels == Y_test); % Count matches

    % Step 4: Calculate accuracy
    total_predictions = length(Y_test); % Total number of data points
    accuracy = correct_predictions / total_predictions;

    % Display the accuracy
    fprintf('Accuracy: %.2f%%\n', accuracy * 100);
end