addpath('./functions');
n = 100;  % Number of samples
c = 5;    % Number of clusters
m = 10;   % Number of features
X = rand(m, n);  % Random data matrix (replace with your actual data)
k = 5;    % Number of neighbors for local cliques

% Call the function
[Y, local_cliques] = initial_clique(n, c, X, k);

% Display results
%disp('Cluster Assignment Matrix Y:');
%disp(Y);

disp('Local Cliques:');
disp(local_cliques);


disp('Check the size of Y:')
% Here Y represents the predict 
disp(size(Y))
G = G_for_Y(Y);
test_G = G'*G;
disp('G:');
disp(size(G));
disp('Test of G^T*G:');
disp(test_G);

% Call the function from ./functions
i = 1;  % Index of the clique (e.g., first clique)
G_i = extract_local_G(G, local_cliques, i);

% Display G_{(i)}
disp('Local Scaled Cluster Assignment Matrix G_{(i)}:');
disp(G_i);

lambda = 0.5;  % Regularization parameter

L_total = 0;
for i=1:n
    L_total = L_total + compute_L_i(X, local_cliques, i, lambda);
end

% Display L_i
disp('Local Laplacian Matrix L_i:');
disp(L_i);
disp(L_total);

[Vectors, Lambdas] = eig(L_total);
disp('Eigenvalues:');
disp(diag(Lambdas));  % Extract diagonal elements of D (eigenvalues)

disp('Eigenvectors:');
disp(Vectors);  % Columns of V are the eigenvectors