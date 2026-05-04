% Computes the configuration all the links of a serial chain mechanism relative to the base frame Psi_0
% Input: unit_twists - cell array of n 6x1 matrices - unit twists of the joints T_i^{(i-1), (i-1)}
%        H0s - cell array of n 4x4 matrices - reference (q = 0) configurations of the links H_i^{i-1}(0)
%        q - nx1 matrix - joint variables q^i
% Output: Hs - cell array of n+1 4x4 matrices - current configurations of the links 0 to n of the serial chain mechanism relative to the base frame Psi_0

function Hs = direct_kinematics(unit_twists, H0s, q)
    n = length(unit_twists);
        Hs = cell(1, n+1);
    % >>>>>> INSERT YOUR CODE HERE <<<<<<
end