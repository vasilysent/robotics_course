function robot()

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
% Create figure and axes for the interactive robot
fig1 = figure('Name', 'Interactive 6-DOF Robot Manipulator', 'NumberTitle', 'off');
ax1 = axes('Parent', fig1, 'Projection', 'perspective'); % Set projection to perspective
hold(ax1, 'on');
grid(ax1, 'on');
axis(ax1, 'equal');
xlabel(ax1, 'X');
ylabel(ax1, 'Y');
zlabel(ax1, 'Z');
view(ax1, [1, 1, 1]);
xlim(ax1, [-500, 500]);
ylim(ax1, [-500, 500]);
zlim(ax1, [-100, 1000]);
axis(ax1, 'manual'); % Fix the axes scaling and limits

% Number of joints
num_joints = length(unit_twists);

% Initialize default q values
q = zeros(num_joints, 1);

% Slider parameters
slider_min = -pi;  % Minimum joint angle
slider_max = pi;   % Maximum joint angle
slider_step = [0.01, 0.10];  % Small and large step sizes

% Store q and other variables in the figure's UserData
fig1.UserData.q = q;
fig1.UserData.unit_twists = unit_twists;
fig1.UserData.H0s = H0s;
fig1.UserData.ax = ax1;

% Create sliders
for i = 1:num_joints
    uicontrol('Parent', fig1, 'Style', 'slider', ...
                'Min', slider_min, 'Max', slider_max, ...
                'Value', q(i), ...
                'SliderStep', slider_step, ...
                'Position', [20, 60 + (i-1)*40, 200, 20], ...
                'Callback', @(src, event) slider_callback(src, event, i));
    % Add a label for the slider
    uicontrol('Parent', fig1, 'Style', 'text', ...
                'Position', [230, 60 + (i-1)*40, 50, 20], ...
                'String', sprintf('q%d', i));
end

% Compute initial Hs and plot the robot
Hs = direct_kinematics(unit_twists, H0s, q);
plot_robot(ax1, Hs);

%% Callback function for sliders
function slider_callback(src, ~, joint_idx)
    % Get current q values
    q = fig1.UserData.q;

    % Update q for the joint that moved
    q(joint_idx) = src.Value;

    % Update q in UserData
    fig1.UserData.q = q;

    % Compute the transformations based on q
    Hs = direct_kinematics(unit_twists, H0s, q);

    % Update the robot plot
    plot_robot(ax1, Hs);
end

end