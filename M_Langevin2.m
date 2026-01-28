%% SCRIPT 20: Langevin Decoupled Analysis (Stiffness vs. Temperature)
% This script tests the "Decoupled Drivers" hypothesis:
% 1. Controls' cognition is driven by Landscape Flatness (Curvature).
% 2. Patients' cognition is driven by System Temperature (Diffusion).
%
% METRICS RECONSTRUCTED FROM DATA:
% - Curvature (b): Steepness of the potential well.
% - Diffusion (D): Magnitude of intrinsic neural noise.

clc; %close all;

% Ensure results from Script 19 exist
if ~exist('results_controls', 'var') || ~exist('results_patients', 'var')
    error('Please run Script 19 first to generate Langevin results.');
end

%% --- 1. Prepare Data ---
% Cols: [WASI, Curvature(b), Diffusion(D), Quartic(a)]
% We use Z-scores for visualization to put them on the same scale, 
% but raw correlations are identical.

all_curve = [results_controls(:,2); results_patients(:,2)];
all_diff  = [results_controls(:,3); results_patients(:,3)];

% Global Normalization (Z-score based on whole dataset)
norm_curve = (all_curve - mean(all_curve)) / std(all_curve);
norm_diff  = (all_diff - mean(all_diff)) / std(all_diff);

% Split back
n_c = size(results_controls, 1);
n_p = size(results_patients, 1);

% Data Matrices: [WASI, Norm_Curvature, Norm_Diffusion]
dat_c = [results_controls(:,1), norm_curve(1:n_c), norm_diff(1:n_c)];
dat_p = [results_patients(:,1), norm_curve(n_c+1:end), norm_diff(n_c+1:end)];

%% --- 2. Plotting (Decoupled Mechanisms) ---
fig = figure('Name', 'Decoupled Drivers of Cognition', 'WindowState', 'maximized');
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- ROW 1: CORRELATION ANALYSIS (Who drives whom?) ---

% Plot 1: Curvature (Stiffness) vs WASI
nexttile;
hold on;
[beta_c_curve, p_c_curve] = plot_correlation_group(dat_c, 2, 'Controls', 'b');
[beta_p_curve, p_p_curve] = plot_correlation_group(dat_p, 2, 'Patients', 'r');
title(sprintf('Driver 1: Landscape Stiffness (Curvature)\nControls: \\beta=%.2f, p=%.3f | Patients: \\beta=%.2f, p=%.3f', ...
    beta_c_curve, p_c_curve, beta_p_curve, p_p_curve));
xlabel('WASI'); ylabel('Potential Curvature (Z)');
legend('Location', 'best');
grid on;

% Plot 2: Diffusion (Temperature) vs WASI
nexttile;
hold on;
[beta_c_diff, p_c_diff] = plot_correlation_group(dat_c, 3, 'Controls', 'b');
[beta_p_diff, p_p_diff] = plot_correlation_group(dat_p, 3, 'Patients', 'r');
title(sprintf('Driver 2: System Temperature (Diffusion)\nControls: \\beta=%.2f, p=%.3f | Patients: \\beta=%.2f, p=%.3f', ...
    beta_c_diff, p_c_diff, beta_p_diff, p_p_diff));
xlabel('WASI'); ylabel('Intrinsic Diffusion (Z)');
% No legend needed (redundant)
grid on;

% --- ROW 2: GROUP COMPARISONS (Boxplots) ---
% Confirm the "Frozen" hypothesis (Patients have lower temp)

% Plot 3: Curvature Extremes
nexttile;
compare_extremes_metric(dat_c, dat_p, 2, 'Stiffness (Curvature)', 10);

% Plot 4: Diffusion Extremes
nexttile;
compare_extremes_metric(dat_c, dat_p, 3, 'Temperature (Diffusion)', 10);


%% --- Helper Functions ---
function [beta, p_val] = plot_correlation_group(data, col_idx, label, color_c)
    x = data(:,1); 
    y = data(:,col_idx);
    
    % Scatter
    scatter(x, y, 60, color_c, 'filled', 'MarkerFaceAlpha', 0.6, 'DisplayName', label);
    
    % Plot Regression Line (On Raw Axes)
    [b, ~] = robustfit(x, y);
    x_grid = linspace(min(x), max(x), 100)';
    y_fit = b(1) + b(2)*x_grid;
    plot(x_grid, y_fit, 'Color', color_c, 'LineWidth', 2, ...
        'DisplayName', sprintf('%s Fit', label));
        
    % Calculate Standardized Robust Statistics for Reporting
    % We Z-score both X and Y LOCALLY to get a correlation-like Beta
    x_z = (x - mean(x)) / std(x);
    y_z = (y - mean(y)) / std(y);
    
    [b_std, stats] = robustfit(x_z, y_z);
    
    beta = b_std(2); % Standardized Beta
    p_val = stats.p(2);
end

function compare_extremes_metric(c, p, col, name, N)
    % Sort by WASI
    [~, ic] = sort(c(:,1)); 
    [~, ip] = sort(p(:,1));
    
    % Extract Top/Bottom
    c_low = c(ic(1:N), col);
    c_high = c(ic(end-N+1:end), col);
    p_low = p(ip(1:N), col);
    p_high = p(ip(end-N+1:end), col);
    
    dat = [c_low; c_high; p_low; p_high];
       
    grp = [repmat({'C-Low'},N,1); repmat({'C-High'},N,1); ...
           repmat({'P-Low'},N,1); repmat({'P-High'},N,1)];
       
    % Plot
    boxplot(dat, grp, 'Colors', 'k', 'Symbol', '+');
    ylabel([name ' (Z)']);
    hold on;
    
    % --- Statistics ---
    % 1. Global Group Difference (C vs P)
    [~, p_global] = ttest2(c(:,col), p(:,col));
    
    % 2. Subgroup Differences
    [p_sub_c, ~, ~] = ranksum(c_low, c_high);
    [p_sub_p, ~, ~] = ranksum(p_low, p_high);
    
    % --- Draw Brackets ---
    y_lims = ylim;
    y_max = y_lims(2);
    y_min = y_lims(1);
    yrange = y_max - y_min;
    offset = yrange * 0.05;
    
    % Bracket for Controls (1 vs 2)
    line([1, 2], [y_max, y_max] + offset, 'Color', 'k', 'LineWidth', 1.5);
    text(1.5, y_max + offset*1.5, sprintf('p=%.3f', p_sub_c), 'HorizontalAlignment', 'center', 'FontSize', 9);
    
    % Bracket for Patients (3 vs 4)
    line([3, 4], [y_max, y_max] + offset, 'Color', 'k', 'LineWidth', 1.5);
    text(3.5, y_max + offset*1.5, sprintf('p=%.3f', p_sub_p), 'HorizontalAlignment', 'center', 'FontSize', 9);
    
    % Adjust Y-limit to fit brackets
    ylim([y_min, y_max + yrange * 0.15]);
    
    title(sprintf('%s\nGlobal Diff C vs P: p=%.4f', name, p_global));
    grid on;
end