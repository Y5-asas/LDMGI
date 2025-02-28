function G = G_for_Y(Y)
    % Calculate the scaled cluster assignment vector
    % Input Y (matrix with size of n*c)
    % Output G (matrix with sized of n*c)
    Y_mult = Y'*Y   ; % Since we know that Y'* Y have only diagonal term, we can use more simple method to calculate
    diag_elements = diag(Y_mult);
    if any(diag_elements <= 0)
    error('Diagonal elements must be positive.');
    end
    % Compute the inverse square root of the diagonal elements
    inv_sqrt_diag = 1 ./ sqrt(diag_elements);
    Y_inv_sqrt = diag(inv_sqrt_diag);
    G = Y*Y_inv_sqrt;
   