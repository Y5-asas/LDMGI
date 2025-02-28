function H_d = H_d_for_x(dimention)
    % THis function calculate the H_d as the literature
    % Input: Dimention d(int)
    % Output: Matrix H_d
    one_d = ones(dimention,1);
    H_d = eye(dimention) - 1/dimention*(one_d*one_d');
