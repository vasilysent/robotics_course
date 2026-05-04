function plot_robot(ax, Hs)
    % Check if ax is a valid axes handle
    if ~ishandle(ax) || ~strcmp(get(ax, 'Type'), 'axes')
        error('Invalid axes handle (programme interrupted)');
    end
    % Clear the axes without resetting properties
    cla(ax);

    % Number of frames (including the base frame)
    num_frames = length(Hs);

    % Extract positions and orientations
    frame_positions = zeros(3, num_frames);
    frame_orientations = cell(num_frames, 1);

    for i = 1:num_frames
        H = Hs{i};
        frame_positions(:, i) = H(1:3, 4);
        frame_orientations{i} = H(1:3, 1:3); % Rotation matrix
    end

    % Extract X, Y, Z coordinates
    X = frame_positions(1, :);
    Y = frame_positions(2, :);
    Z = frame_positions(3, :);

    % Plot the links
    plot3(ax, X, Y, Z, '-o', 'LineWidth', 10, 'MarkerSize', 6, 'MarkerFaceColor', 'red');

    % Plot coordinate frames
    scale = 100; % Adjust scale as needed
    plotCoordinateFrame(ax, frame_positions(:, 1), frame_orientations{1}, scale); % Base frame
    plotCoordinateFrame(ax, frame_positions(:, num_frames), frame_orientations{num_frames}, scale); % End-effector frame

    % Keep the axis limits fixed
    xlim(ax, [-500, 500]);
    ylim(ax, [-500, 500]);
    zlim(ax, [-100, 1000]);

    % Set labels and grid
    xlabel(ax, 'X');
    ylabel(ax, 'Y');
    zlabel(ax, 'Z');
    grid(ax, 'on');
    axis(ax, 'equal');

    % No need to restore view since we're not changing it
end
