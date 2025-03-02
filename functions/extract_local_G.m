function G_i = extract_local_G(G, local_cliques, i)
    % INPUTS:
    %   G: Scaled cluster assignment matrix of size (n x c)
    %   local_cliques: Cell array of size (n x 1), where each cell contains
    %                  the indices of the k-nearest neighbors for each point
    %   i: Index of the clique for which to extract G_{(i)}
    %
    % OUTPUT:
    %   G_i: Local scaled cluster assignment matrix for the i-th clique

    % Step 1: Get the indices of the local clique
    clique_indices = local_cliques{i};  % Indices of the k-nearest neighbors

    % Step 2: Extract the rows of G corresponding to the clique indices
    G_i = G(clique_indices, :);  % Local scaled cluster assignment matrix
end