% Computes the joint velocities dq to move the end effector to the setpoint
% Input: q - nx1 matrix - joint variables q^i
%        setpoint - 3x1 matrix - desired end-effector position
%        dsetpoint - 3x1 matrix - desired end-effector velocity
%        H0s - cell array of n 4x4 matrices - reference (q = 0) configurations of the links H_i^{i-1}(0)
%        unit_twists - cell array of n 6x1 matrices - unit twists of the joints T_i^{(i-1), (i-1)}
% Output: dq - nx1 matrix - joint velocities dq^i

function dq = calculate_dq(q, setpoint, dsetpoint, H0s, unit_twists)

% >>>>>>>>>>>> INSERT YOUR CODE HERE <<<<<<<<<<<<

% Limit the end-effector velocity
if norm(dp_ee) > 700
    dp_ee = 700*(dp_ee/norm(dp_ee)); 
end

% >>>>>>>>>>>> MODIFY >>>>>>>>>>>>
dq = zeros(size(q));
% <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

end