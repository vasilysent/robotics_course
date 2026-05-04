function robot_control()

%% Robot parameters
a = 90; b = 425; c = 392; d = 93; e = 95; f = 82;


% >>>> TO MODIFY >>>>>>>>
% Unit twists of the joints T_i^{(i-1), (i-1)}
T1 = [0; 0; 0; 0; 0; 0];
T2 = [0; 0; 0; 0; 0; 0];
T3 = [0; 0; 0; 0; 0; 0];
T4 = [0; 0; 0; 0; 0; 0];
T5 = [0; 0; 0; 0; 0; 0];
T6 = [0; 0; 0; 0; 0; 0];

% Reference configurations of the joints H_i^{i-1}(0)
H0_1 = eye(4);
H0_2 = eye(4);
H0_3 = eye(4);
H0_4 = eye(4);
H0_5 = eye(4);
H0_6 = eye(4);
% <<<<<<<<<<<<<<<<<<<<

unit_twists = {T1, T2, T3, T4, T5, T6};
H0s = {H0_1, H0_2, H0_3, H0_4, H0_5, H0_6};


%%
% Control Loop Parameters
T = 0.02;       % Time step
t_end = 20;     % Total simulation time
N = round(t_end / T);  % Number of iterations
t = 0;          % Initialize time vector
q = zeros(6, 1);  % Initialize joint variables

% Create figure and axes for the control loop
fig2 = figure('Name', 'Control 6-DOF Robot Manipulator', 'NumberTitle', 'off');
ax2 = axes('Parent', fig2, 'Projection', 'perspective'); % Set projection to perspective
hold(ax2, 'on');
grid(ax2, 'on');
axis(ax2, 'equal');
xlabel(ax2, 'X');
ylabel(ax2, 'Y');
zlabel(ax2, 'Z');
title(ax2, 'Robot Trajectory Tracking');
view(ax2, [1, 1, 1]);
xlim(ax2, [-500, 500]);
ylim(ax2, [-500, 500]);
zlim(ax2, [-100, 1000]);
axis(ax2, 'manual');  % Fix axis limits

% Before the control loop
trajectory = [];  % Initialize trajectory array

% Main control loop
for i = 1:N
    % Current time
    t = (i - 1) * T;

    % Generate setpoint trajectory
    x = 600;
    y = 200 * cos(t);
    z = 400 + 200 * sin(2*t);
    setpoint = [x; y; z];

    % Compute desired velocities (if applicable)
    dx = 0;
    dy = -200 * sin(t);
    dz = 400 * cos(2*t);
    dsetpoint = [dx; dy; dz];

    % Store the trajectory
    trajectory = [trajectory, setpoint];

    % Calculate desired joint velocities
    dq = calculate_dq(q, setpoint, dsetpoint, H0s, unit_twists);

    % Update joint variables
    q = q + T * dq;

    % Compute the transformations for the current q
    Hs = direct_kinematics(unit_twists, H0s, q);

    % Plot the robot
    plot_robot(ax2, Hs);

    % Plot the desired end-effector position
    plot3(ax2, setpoint(1), setpoint(2), setpoint(3), 'kx', 'MarkerSize', 10, 'LineWidth', 2);

    % Plot the trajectory
    plot3(ax2, trajectory(1, :), trajectory(2, :), trajectory(3, :), 'b--');

    % Update the title with the current time
    title(ax2, sprintf('Robot Trajectory Tracking (t = %.2f s)', t));

    drawnow;
    pause(T);  % Control the loop timing
end

hold(ax2, 'off');

end