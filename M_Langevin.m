%% SCRIPT 19: Data-Driven Langevin Dynamics (Dynamic Global Potential)
% This script reconstructs the brain's Global Dynamical Potential Landscape
% directly from the DYNAMIC Smoothness time series.
%
% CONFIGURATION: 
% - Fixed 20 Bins.
% - Global Stiffness (-p3/2).
% - Central Temperature (Noise near Z=0).
%   (Reverted to Fixed Z=0 approach for stability vs noisy well bottoms).
%
% OUTPUTS:
% 1. Global Stiffness (Robust Curvature)
% 2. Global Diffusion (Intrinsic Noise)
% 3. Central Diffusion (Noise at Equilibrium)

clc; %close all;

%% --- 1. Setup Parameters ---
dt = 1/100; % 10 ms
num_bins = 20; 
N_extremes = 10;

groups_to_analyze = {struct('Name', 'Controls', 'LabelIndex', 0), struct('Name', 'Patients', 'LabelIndex', 1)};
results_controls = [];
results_patients = [];

% Storage for Average Potentials
potentials_c_high = []; potentials_c_low = [];
potentials_p_high = []; potentials_p_low = [];
x_axis_common = linspace(-3, 3, num_bins); 

%% --- 2. Main Subject Loop ---
disp('Reconstructing Dynamic Global Langevin Potentials...');

for subi = 1:nsubj
    if isnan(wasi(subi)), continue; end
    
    try
        % Load Dynamic Graph
        A_dyn = WC_TDA(:,:,:,subi);
        % Load Signal
        elects = squeeze(all_elects(:,:,:,subi));
        
        [N, ~, n_trials] = size(A_dyn);
        samples_per_trial = size(elects, 2);
        
        % FIX: Correctly calculate total samples and reshape 3D -> 2D
        n_samples_total = samples_per_trial * n_trials;
        elects_2d = reshape(elects, N, n_samples_total);
    catch
        continue;
    end
    
    % Signal Power vector [N x Time]
    x_t_all = real(elects_2d).^2;
    
    % Pre-allocate Smoothness Time Series
    smooth_ts = nan(1, n_samples_total);
    
    % Calculate Dynamic Global Smoothness
    for tr = 1:n_trials
        A = A_dyn(:,:,tr);
        if any(isnan(A(:))) || all(A(:)==0), continue; end
        
        D = diag(sum(A, 2));
        L = eye(N) - diag(1./sqrt(diag(D) + eps)) * A * diag(1./sqrt(diag(D) + eps));
        
        idx_start = (tr-1)*samples_per_trial + 1;
        idx_end = tr*samples_per_trial;
        
        x_chunk = x_t_all(:, idx_start:idx_end);
        
        % S(t) = x'Lx (Vectorized diagonal of x'Lx)
        Lx = L * x_chunk;
        smooth_chunk = sum(x_chunk .* Lx, 1);
        
        smooth_ts(idx_start:idx_end) = smooth_chunk;
    end
    
    % --- Kramers-Moyal Reconstruction ---
    smooth_ts = smooth_ts(~isnan(smooth_ts));
    if length(smooth_ts) < 1000, continue; end
    
    % Z-Score (Global Normalization)
    x = (smooth_ts - mean(smooth_ts)) / std(smooth_ts);
    dx = diff(x);
    
    % Robust Adaptive Binning
    limits = prctile(x, [2.5, 97.5]);
    if abs(limits(2)-limits(1)) < 1e-6, limits = [-3, 3]; end
    edges = linspace(limits(1), limits(2), num_bins+1);
    centers = (edges(1:end-1) + edges(2:end)) / 2;
    
    [~, bin_idx] = histc(x(1:end-1), edges);
    
    Drift = nan(1, num_bins);
    Diffusion = nan(1, num_bins);
    
    for b = 1:num_bins
        mask = (bin_idx == b);
        if sum(mask) > 10
            Drift(b) = mean(dx(mask)) / dt;
            Diffusion(b) = var(dx(mask)) / dt;
        end
    end
    
    valid = ~isnan(Drift);
    
    if sum(valid) >= 4
        % 1. Fit Polynomial Force (Stiffness)
        [p, ~] = polyfit(centers(valid), Drift(valid), 3);
        b_quad  = -p(3)/2; 
        
        % 2. Global Diffusion (Average over all valid bins)
        global_diff = mean(Diffusion(valid));
        
        % 3. Central Diffusion (Noise near Equilibrium x=0)
        % Since data is Z-scored, the statistical mean is 0. 
        % We define the "Central Basin" as the range [-1, 1] SD.
        % This avoids noise from finding the exact numerical minimum.
        center_mask = valid & (centers >= -1) & (centers <= 1);
        
        if sum(center_mask) > 0
            central_diff = mean(Diffusion(center_mask));
        else
            central_diff = global_diff; % Fallback
        end
        
        % Store Results: [WASI, Stiffness, Global_Diff, Central_Diff]
        row_data = [wasi(subi), b_quad, global_diff, central_diff];
        
        % Interpolate U for plotting (optional visualization)
        U_curve = -cumsum(Drift(valid)) * (centers(2)-centers(1));
        U_curve = U_curve - min(U_curve);
        valid_centers = centers(valid);
        U_interp = interp1(valid_centers, U_curve, x_axis_common, 'linear', NaN);
        
        if origLabels(subi) == 0
            results_controls = [results_controls; row_data];
            if wasi(subi) > median(wasi(origLabels==0))
                potentials_c_high = [potentials_c_high; U_interp];
            else
                potentials_c_low = [potentials_c_low; U_interp];
            end
        else
            results_patients = [results_patients; row_data];
            if wasi(subi) > median(wasi(origLabels==1))
                potentials_p_high = [potentials_p_high; U_interp];
            else
                potentials_p_low = [potentials_p_low; U_interp];
            end
        end
    end
end

disp('Reconstruction Complete.');

%% --- 3. Plotting ---
fig = figure('Name', 'Global Dynamic Langevin Potential', 'WindowState', 'maximized');
t = tiledlayout(2, 3, 'TileSpacing', 'compact');

% ROW 1: STIFFNESS (Curvature)
nexttile; plot_corr_robust(results_controls(:,1), results_controls(:,2), 'Controls', 'Rigidity', 'b'); yline(0, '--k');
nexttile; plot_corr_robust(results_patients(:,1), results_patients(:,2), 'Patients', 'Rigidity', 'r'); yline(0, '--k');
nexttile; compare_groups_boxplot(results_controls(:,2), results_patients(:,2), 'Rigidity');

% ROW 2: CENTRAL TEMPERATURE (Fixed Z=0 Center)
nexttile; plot_corr_robust(results_controls(:,1), results_controls(:,4), 'Controls', 'Noise', 'b');
nexttile; plot_corr_robust(results_patients(:,1), results_patients(:,4), 'Patients', 'Noise', 'r');
nexttile; compare_groups_boxplot(results_controls(:,4), results_patients(:,4), 'Noise');

%% --- Helper Functions ---
function plot_corr_robust(x, y, name, metric, color_c)
    scatter(x, y, 50, color_c, 'filled', 'MarkerFaceAlpha', 0.6); hold on; 
    [b, stats] = robustfit(x, y);
    x_grid = linspace(min(x), max(x), 100)';
    plot(x_grid, b(1)+b(2)*x_grid, 'Color', color_c, 'LineWidth', 2);
    title(sprintf('%s: %s\nRobust \\beta=%.2f, p=%.3f', name, metric, b(2), stats.p(2)));
    xlabel('WASI'); ylabel(metric); grid on;
end

function compare_groups_boxplot(c_data, p_data, metric_name)
    data = [c_data; p_data];
    groups = [repmat({'Controls'}, size(c_data,1), 1); repmat({'Patients'}, size(p_data,1), 1)];
    boxplot(data, groups, 'Colors', 'br', 'Symbol', '+'); ylabel(metric_name);
    [p_val, ~, ~] = ranksum(c_data, p_data);
    if p_val > 0.05, sig_str = 'ns'; else, sig_str = '*'; end
    title(sprintf('Group Comparison\n%s (p=%.3f)', sig_str, p_val)); grid on;
end