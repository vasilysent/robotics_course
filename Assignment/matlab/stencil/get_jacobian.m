% Computes the geometric Jacobian matrix of a serial chain mechanism
% Input: unit_twists - cell array of n 6x1 matrices - unit twists of the joints T_i^{(i-1), (i-1)}
%        H0s - cell array of (at least) n-1 4x4 matrices - reference configurations of the joints H_i^{i-1}(0)
%        q - nx1 matrix - joint variables q^i

function J = get_jacobian(unit_twists, H0s, q)
    n = length(unit_twists);
    % >>>>>> MODIFY >>>>>>
    J = zeros(6, n);
    % <<<<<<<<<<<<<<<<<<<<
end