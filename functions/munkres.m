function assignment = munkres(cost)
    % Implementation of the Hungarian algorithm for optimal assignment.
    % cost: Cost matrix (n x n).
    % assignment: Optimal assignment vector (1 x n).

    n = size(cost, 1);
    cost = double(cost); % Ensure cost is of type double

    % Step 1: Subtract the minimum of each row from the row
    cost = cost - min(cost, [], 2);

    % Step 2: Subtract the minimum of each column from the column
    cost = cost - min(cost, [], 1);

    % Step 3: Find a zero in the cost matrix and mark it as assigned
    [assignment, ~] = hungarian_algorithm(cost);
end

function [assignment, cost] = hungarian_algorithm(cost)
    % Core implementation of the Hungarian algorithm.
    n = size(cost, 1);
    assignment = zeros(1, n); % Initialize assignment vector
    cost = double(cost); % Ensure cost is of type double

    % Step 4: Cover all zeros with a minimum number of lines
    [row_covered, col_covered, marked_zeros] = cover_zeros(cost);

    while sum(row_covered) + sum(col_covered) < n
        % Step 5: Find the minimum uncovered element
        uncovered_cost = cost(~row_covered, ~col_covered);
        min_uncovered = min(uncovered_cost(:));

        % Step 6: Subtract the minimum from uncovered elements and add it to doubly covered elements
        cost(~row_covered, :) = cost(~row_covered, :) - min_uncovered;
        cost(:, ~col_covered) = cost(:, ~col_covered) + min_uncovered;

        % Step 7: Update the coverage and marked zeros
        [row_covered, col_covered, marked_zeros] = cover_zeros(cost);
    end

    % Step 8: Extract the optimal assignment
    assignment = marked_zeros;
end

function [row_covered, col_covered, marked_zeros] = cover_zeros(cost)
    % Cover all zeros in the cost matrix with a minimum number of lines.
    n = size(cost, 1);
    row_covered = false(1, n);
    col_covered = false(1, n);
    marked_zeros = zeros(1, n);

    % Mark rows and columns with zeros
    for i = 1:n
        for j = 1:n
            if cost(i, j) == 0 && ~row_covered(i) && ~col_covered(j)
                marked_zeros(i) = j;
                row_covered(i) = true;
                col_covered(j) = true;
            end
        end
    end
end