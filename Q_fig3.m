%% SCRIPT 25: Full In-Silico Pharmacology Grid (Langevin)
% This script applies Langevin Inverse Modeling to the full 3x3 grid of 
% biophysical simulations.
%
% CONFIGURATION:
% - Central Temperature (Noise near Equilibrium Z=0).
% - Fixed 10 Bins (Robust for short simulation data).
%
% GOAL: Demonstrate that Anti-Seizure Medications (VPA, LEV) universally 
% drive the network towards a "Rigid" (High Stiffness) state.

clc; %close all;

%% --- 1. Setup Parameters ---
dt_sim = 1/100; % 100 Hz
num_bins = 20;  % Reduced bin count for robust simulation estimation
x_axis_common = linspace(-3, 3, num_bins);

conditions = {'GABA', 'Micro', 'Arbor'};
treatments = {'Untreated', 'VPA', 'LEV'};

% Plotting Colors (Accessible Palette)
c_untreated = [0.5 0.5 0.5]; % Grey (Neutral)
c_vpa = [0.0 0.45 0.74];       % Blue (Accessible)
c_lev = [0.85 0.33 0.10];      % Orange (Accessible)
color_map = [c_untreated; c_vpa; c_lev];

% Storage [Conditions x Treatments]
res_stiff = nan(3, 3);
res_temp = nan(3, 3);

%% --- 2. Main Analysis Loop ---
disp('Analyzing 3x3 Pharmacology Grid...');

for c = 1:length(conditions)
    for t = 1:length(treatments)
        cond = conditions{c};
        treat = treatments{t};
        
        filename = sprintf('InSilico_%s_%s.mat', cond, treat);
        
        if ~isfile(filename)
            warning('File %s not found. Run Python script first.', filename);
            continue;
        end
        
        fprintf('Processing %s - %s...\n', cond, treat);
        data = load(filename);
        
        A = double(data.A_dyn);
        X = double(data.elects);
        [N, ~] = size(X);
        
        % --- Smoothness Calculation ---
        A(isnan(A)) = 0;
        D_mat = diag(sum(A, 2));
        d_inv = 1./sqrt(diag(D_mat) + eps);
        L = eye(N) - diag(d_inv) * A * diag(d_inv);
        
        P = X.^2;
        % Global Smoothness S(t) = sum(P .* (L*P))
        smooth_ts = sum(P .* (L * P), 1);
        
        % --- Langevin Reconstruction (Robust) ---
        x_state = (smooth_ts - mean(smooth_ts)) / std(smooth_ts);
        dx = diff(x_state);
        
        % Adaptive Binning (2.5 - 97.5 percentile)
        limits = prctile(x_state, [2.5, 97.5]);
        if abs(limits(2)-limits(1)) < 0.1, limits = [-3, 3]; end
        
        edges = linspace(limits(1), limits(2), num_bins+1);
        centers = (edges(1:end-1) + edges(2:end)) / 2;
        
        [~, bin_idx] = histc(x_state(1:end-1), edges);
        
        Drift = nan(1, num_bins);
        Diffusion = nan(1, num_bins);
        
        for b = 1:num_bins
            mask = (bin_idx == b);
            if sum(mask) > 10 % Threshold for valid bin
                Drift(b) = mean(dx(mask)) / dt_sim;
                Diffusion(b) = var(dx(mask)) / dt_sim;
            end
        end
        
        valid = ~isnan(Drift);
        if sum(valid) < 4
            warning('  Not enough valid bins.');
            continue;
        end
        
        % Fit Polynomial: F(x) = -ax^3 - bx - c
        [p_poly, ~] = polyfit(centers(valid), Drift(valid), 3);
        
        % Extract Metrics
        b_stiffness = -p_poly(3)/2; % Curvature
        
        % Central Temperature (Noise near Equilibrium x=0)
        % Consistent with Clinical Analysis: Fixed window [-1, 1] SD
        center_mask = valid & (centers >= -1) & (centers <= 1);
        
        if sum(center_mask) > 0
            d_temp = mean(Diffusion(center_mask));
        else
            d_temp = mean(Diffusion(valid)); % Fallback to global
        end
        
        res_stiff(c, t) = b_stiffness;
        res_temp(c, t) = d_temp;
    end
end

%% --- 3. Visualization (Publication Figure) ---
if all(isnan(res_stiff(:))), return; end

fig = figure('Name', 'Pharmacological Thermodynamic Shift', 'Color', 'w', 'Position', [100, 100, 1200, 700]);
tlay = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Titles for columns
cond_titles = {'GABA Deficit', 'Microdysgenesis', 'Poor Arborization'};

for c = 1:3
    % --- Row 1: Rigidity (Stiffness) ---
    nexttile(c);
    hold on;
    % Plot each bar individually to control texture/color
    for k = 1:3
        b_plot = bar(k, res_stiff(c, k), 'FaceColor', color_map(k,:), 'EdgeColor', 'k');
        % Apply Texture (Function below)
        apply_texture(k, res_stiff(c, k), 0.8, k);
    end
    
    ylabel('Stiffness (b)');
    title([cond_titles{c} ' - Rigidity']);
    xticks(1:3); xticklabels(treatments); xtickangle(45);
    
    % Expand Y-limits to fit text
    curr_data = res_stiff(c, :);
    y_range = max(curr_data) - min(curr_data);
    if y_range == 0, y_range = abs(curr_data(1)); end
    if y_range == 0, y_range = 1; end
    ylim([min(0, min(curr_data)*1.1), max(curr_data) + 0.25*y_range]);
    
    grid on;
    
    % Add percentage change text
    base_val = res_stiff(c, 1);
    for k = 2:3
        curr_val = res_stiff(c, k);
        pct_change = ((curr_val - base_val) / abs(base_val)) * 100;
        
        % Position text slightly above bar
        text(k, curr_val, sprintf('%+.0f%%', pct_change), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', 9, 'FontWeight', 'bold');
    end
    
    % --- Row 2: Noise (Temperature) ---
    nexttile(c+3);
    hold on;
    for k = 1:3
        b_plot2 = bar(k, res_temp(c, k), 'FaceColor', color_map(k,:), 'EdgeColor', 'k');
        apply_texture(k, res_temp(c, k), 0.8, k);
    end
    
    ylabel('Temperature (D)');
    title([cond_titles{c} ' - Noise']);
    xticks(1:3); xticklabels(treatments); xtickangle(45);
    
    % Expand Y-limits for Temperature too
    curr_temp = res_temp(c, :);
    t_range = max(curr_temp) - min(curr_temp);
    if t_range == 0, t_range = abs(curr_temp(1)); end
    ylim([0, max(curr_temp) + 0.2*t_range]);
    
    grid on;
end

% --- Global Legend ---
% Create invisible dummy plots to generate a correct legend for the colors/textures
hold on;
h_leg = [];
for k = 1:3
    % Create a dummy bar object for the legend
    h_leg(k) = bar(nan, nan, 'FaceColor', color_map(k,:), 'EdgeColor', 'k');
end

lg = legend(h_leg, treatments, 'Orientation', 'horizontal', 'Box', 'off');
lg.Layout.Tile = 'south';


%% --- Helper Function for Texture ---
function apply_texture(x, y, w, type)
    % Adds simple line-based texture to bars for accessibility
    % type 1: Untreated (Solid - No texture)
    % type 2: VPA (Horizontal Stripes)
    % type 3: LEV (Dots/Stipple)
    
    if isnan(y) || y==0, return; end
    
    x_left = x - w/2;
    x_right = x + w/2;
    y_base = 0; % Assuming bars start at 0
    
    % Determine range
    y_min = min(y_base, y);
    y_max = max(y_base, y);
    
    hold on;
    
    if type == 2 % VPA: Horizontal Stripes
        num_stripes = 6;
        for i = 1:num_stripes
            y_line = y_min + (i/(num_stripes+1)) * (y_max - y_min);
            plot([x_left, x_right], [y_line, y_line], 'Color', [1 1 1 0.6], 'LineWidth', 1.5);
        end
        
    elseif type == 3 % LEV: Dots
        num_rows = 4;
        num_cols = 3;
        x_step = w / (num_cols + 1);
        y_step = (y_max - y_min) / (num_rows + 1);
        
        for i = 1:num_cols
            for j = 1:num_rows
                px = x_left + i*x_step;
                py = y_min + j*y_step;
                plot(px, py, 'w.', 'MarkerSize', 8);
            end
        end
    end
    % Type 1 (Untreated) is left solid
end