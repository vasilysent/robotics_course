    function plotCoordinateFrame(ax, origin, rotationMatrix, scale)
        x_axis = rotationMatrix(:, 1) * scale;
        y_axis = rotationMatrix(:, 2) * scale;
        z_axis = rotationMatrix(:, 3) * scale;

        % Plot X-axis (Red)
        quiver3(ax, origin(1), origin(2), origin(3), x_axis(1), x_axis(2), x_axis(3), 'r', 'LineWidth', 2);
        % Plot Y-axis (Green)
        quiver3(ax, origin(1), origin(2), origin(3), y_axis(1), y_axis(2), y_axis(3), 'g', 'LineWidth', 2);
        % Plot Z-axis (Blue)
        quiver3(ax, origin(1), origin(2), origin(3), z_axis(1), z_axis(2), z_axis(3), 'b', 'LineWidth', 2);
    end
