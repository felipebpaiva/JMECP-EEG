%% SCRIPT 9: Plot Dynamic Feature Heatmap by WASI
% This script loads all dynamic features (std, mafd, sampen for 0D, 1D, and AP)
% and visualizes them as a heatmap.
%
% Subjects (rows) are sorted by their WASI score, and features (columns)
% are z-scored across subjects to make them comparable.
% This creates a "composite" horizontal bar effect for each subject.

%clear all;
clc;

%% --- 1. Load All Necessary Data ---
% disp('Loading feature and behavioral data...');
% try
%     load('K:\JMECP_EEG_Analysis\results\RELAXv2\Preprocessed_GED_Graphs.mat', 'origLabels', 'wasi');
%     load('K:\JMECP_EEG_Analysis\results\RELAXv2\Topology_Features.mat', 'dynamic_features_0D', 'dynamic_features_1D');
%     load('K:\JMECP_EEG_Analysis\results\RELAXv2\AlphaPower_Features.mat', 'dynamic_features_AP');
% catch ME
%     disp('Error loading .mat files. Please ensure you have run previous scripts (02, 03) and saved their outputs.');
%     rethrow(ME);
% end
disp('Data loaded.');

%% --- 2. Define Features and Colormap ---
    
% Build feature list from all 3 sources
feature_list = {
    'smoothness_std_0D', 'smoothness_mafd_0D', 'smoothness_sampen_0D', ...
    'smoothness_std_1D', 'smoothness_mafd_1D', 'smoothness_sampen_1D', ...
    'smoothness_std_AP', 'smoothness_mafd_AP', 'smoothness_sampen_AP'
    };
    
% Create the custom Red-White-Blue colormap (Blue=High, Red=Low)
bwr_cmap = create_rwb_colormap();

%% --- 3. Process and Plot Data for Each Group ---

groups_to_analyze = {struct('Name', 'Controls', 'LabelIndex', 0), struct('Name', 'Patients', 'LabelIndex', 1)};

for g = 1:length(groups_to_analyze)
    current_group = groups_to_analyze{g};
    
    % --- 3A. Prepare Data for this Group ---
    group_idx = find(origLabels == current_group.LabelIndex);
    wasi_group = wasi(group_idx);
    
    num_subjects_group = length(group_idx);
    num_features = length(feature_list);
    feature_matrix = nan(num_subjects_group, num_features);
    
    % Build the feature matrix for this group
    for f = 1:num_features
        feature_name = feature_list{f};
        if contains(feature_name, '_0D')
            source = dynamic_features_0D;
            name_key = strrep(feature_name, '_0D', '');
        elseif contains(feature_name, '_1D')
            source = dynamic_features_1D;
            name_key = strrep(feature_name, '_1D', '');
        elseif contains(feature_name, '_AP')
            source = dynamic_features_AP;
            name_key = strrep(feature_name, '_AP', '');
        end
        
        if isfield(source, name_key)
            feature_data = source.(name_key);
            if isrow(feature_data)
                 feature_matrix(:, f) = feature_data(group_idx)';
            else
                 feature_matrix(:, f) = feature_data(group_idx);
            end
        end
    end
    
    % Step 1 - Mean Imputation for NaN values
    imputed_matrix = feature_matrix;
    for f = 1:num_features
        col_data = imputed_matrix(:, f);
        col_mean = mean(col_data, 'omitnan');
        if isnan(col_mean), col_mean = 0; end % Handle all-NaN column
        col_data(isnan(col_data)) = col_mean;
        imputed_matrix(:, f) = col_data;
    end
    
    % Step 2 - Normalize each feature column 0-100 (Min-Max Scaling)
    normalized_matrix = nan(size(imputed_matrix));
    for f = 1:num_features
        col_data = imputed_matrix(:, f);
        min_val = min(col_data);
        max_val = max(col_data);
        range_val = max_val - min_val;
        
        if range_val < eps % Handle constant column (avoid division by zero)
            normalized_matrix(:, f) = 50; % Assign mid-point
        else
            normalized_matrix(:, f) = 100 * (col_data - min_val) / range_val;
        end
    end
    
    % Calculate the composite score (average of 0-100 normalized features)
    composite_score = mean(normalized_matrix, 2);
    
    % Sort by WASI score
    [wasi_sorted, sort_idx] = sort(wasi_group);
    composite_score_sorted = composite_score(sort_idx);
    
    % --- 3B. Plot Horizontal Bar Chart ---
    fig = figure('Name', [current_group.Name ': Composite Dynamic Score by WASI'], 'WindowState', 'maximized');
    ax = axes('Parent', fig);
    
    b = barh(composite_score_sorted);
    hold(ax, 'on');
    
    % --- 3C. Color-code Bars by WASI Score ---
    wasi_limits = [min(wasi, [], 'omitnan'), max(wasi, [], 'omitnan')];
    colors = scale_to_colormap(wasi_sorted, wasi_limits, bwr_cmap);
    
    b.FaceColor = 'flat'; % Enable individual bar coloring
    for i = 1:num_subjects_group
        if ~isnan(composite_score_sorted(i))
            b.CData(i, :) = colors(i, :);
        end
    end
    
    % --- 3D. Add Moving Average Trend Line ---
    % Calculate a moving average of the composite score
    window_size = round(num_subjects_group / 4); % Use a window ~20% of the data
    if window_size < 3, window_size = 3; end % Ensure window is at least 3
    trend_line = smoothdata(composite_score_sorted, 'movmean', window_size);
    
    % Plot the trend line on top
    plot_y_axis = 1:num_subjects_group;
    p = plot(ax, trend_line, plot_y_axis, 'k-', 'LineWidth', 3);
    
    hold(ax, 'off');
    
    % --- 3E. Format Axes ---
    xlabel(ax, 'Composite Dynamic Score (0-100 Scale)');
    ylabel(ax, 'Subjects (Sorted by WASI)');
    
    % Create Y-tick labels with WASI scores
    ytick_pos = 1:num_subjects_group;
    ytick_lab = cell(num_subjects_group, 1);
    for i = 1:num_subjects_group
        ytick_lab{i} = sprintf('Subj %d (WASI: %.0f)', i, wasi_sorted(i));
    end
    
    % Show all Y-ticks
    set(ax, 'YTick', ytick_pos) %, 'YTickLabel', ytick_lab);
    if num_subjects_group > 40
        set(ax, 'FontSize', 8);
    end
    set(ax, 'YLim', [0.5, num_subjects_group + 0.5]);
    
    % Add colorbar and legend
    cb = colorbar(ax);
    cb.Label.String = 'WASI Score';
    colormap(ax, bwr_cmap);
    clim(ax, wasi_limits);
    
    legend(ax, [b, p], {'Subject Composite Score', 'Moving Average Trend'}, 'Location', 'southeast');
    
    % Use robust method for main title
    ax_title = axes('Parent', fig, 'Position', [0 0 1 1], 'Visible', 'off');
    title_text = [current_group.Name ': Smoothness Composite Dynamic Score (Sorted by WASI)'];
    text(0.5, 0.97, title_text, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 16, 'FontWeight', 'bold', 'Parent', ax_title);
    
end

disp('Plotting complete.');


%% --- Helper Function: Red-White-Blue Colormap ---
function cmap = create_rwb_colormap()
    % Creates a diverging colormap: Red (low) -> White (mid) -> Blue (high)
    % As requested: Blue for HIGH WASI, Red for LOW WASI.
    
    % Red (low values)
    r1 = [0.6, 0, 0];
    % White (zero)
    w = [1, 1, 1];
    % Blue (high values)
    b1 = [0, 0, 0.6];
    
    n = 128; % Number of points on each side
    
    % Interpolate
    red_half = [linspace(r1(1), w(1), n)', linspace(r1(2), w(2), n)', linspace(r1(3), w(3), n)'];
    % **FIX**: Corrected 'bf(2)' typo
    blue_half = [linspace(w(1), b1(1), n)', linspace(w(2), b1(2), n)', linspace(w(3), b1(3), n)'];
    
    % Combine (remove duplicate white)
    cmap = [red_half; blue_half(2:end, :)];
end

%% --- Helper Function for Colormap Scaling ---
function colors = scale_to_colormap(values, c_limits, cmap)
    % Scales a vector of values to a colormap
    norm_values = (values - c_limits(1)) / (c_limits(2) - c_limits(1));
    norm_values(norm_values < 0) = 0;
    norm_values(norm_values > 1) = 1;
    cmap_indices = round(norm_values * (size(cmap, 1) - 1)) + 1;
    
    nan_indices = isnan(values);
    colors = zeros(length(values), 3);
    
    valid_indices = ~nan_indices;
    % Handle potential edge case where all indices are NaN
    if ~isempty(valid_indices) && ~isempty(cmap_indices(valid_indices))
        colors(valid_indices, :) = cmap(cmap_indices(valid_indices), :);
    end
    colors(nan_indices, :) = repmat([0.5 0.5 0.5], sum(nan_indices), 1);
end

