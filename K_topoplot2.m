%% SCRIPT 10: Plot Energy Maps for WASI Extremes
% This script tests the hypothesis that subjects at the extremes of the
% WASI score distribution (Top 10 vs. Bottom 10) have different spatial
% distributions of GSP energy in the Q1 (coupled) and Q4 (decoupled) bands.
%
% It generates two figures (Controls, Patients), each with a 2x2 layout
% showing the Q1/Q4 energy maps for the Top 10 vs. Bottom 10 performers.

%clear all;
clc;

%% --- 1. Load All Necessary Data ---
disp('Loading feature and behavioral data...');
try
    % Load the full time series
    load('K:\JMECP_EEG_Analysis\results\RELAXv2\Preprocessed_GED_timeseries_8to10hz.mat')
    load('K:\JMECP_EEG_Analysis\results\RELAXv2\Preprocessed_GED_Graphs_3s_8to10hz.mat', ...
         'origLabels', 'subcode', 'nsubj', 'Nroi', 'ntrials', 'EEG','ciPLV_global');
    % Load the global graphs and WASI scores
    load('K:\JMECP_EEG_Analysis\results\RELAXv2\all_features_3s_8to10hz.mat', 'wasi');
catch ME
    disp('Error loading .mat files. Please ensure you have run previous scripts.');
    rethrow(ME);
end

% Ensure WASI scores match the loaded subjects (if 'all_elects' was pre-cleaned)
if length(wasi) ~= nsubj
    error('Mismatch')
end
disp('Data loaded.');

%% --- 2. Define Parameters ---
N_extremes = 10;
q1_indices = 1:round(Nroi * 0.25);
q4_indices = round(Nroi * 0.75):Nroi;
groups_to_analyze = {struct('Name', 'Controls', 'LabelIndex', 0), struct('Name', 'Patients', 'LabelIndex', 1)};
total_timepoints = ntrials * size(all_elects,2);
% **NEW**: Create time vector in seconds
resfreq = EEG.srate;
time_vec_sec = (1:total_timepoints) / resfreq;
spike_threshold_sd = 5;

%% --- 3. Calculate All Energy Maps and Time Series First ---
disp('Calculating all energy maps and time series for all groups and extremes...');

all_q1_topomaps = cell(2, 2); % (Group, Extreme)
all_q4_topomaps = cell(2, 2);
all_q1_timeseries = cell(2, 2);
all_q4_timeseries = cell(2, 2);
all_smoothness_ts_avg = cell(2, 2); 
all_smoothness_ts_ind = cell(2, 2); % Store individual smoothness ts

for g = 1:length(groups_to_analyze)
    current_group = groups_to_analyze{g};
    
    % Find WASI Extremes
    group_idx_mask = (origLabels == current_group.LabelIndex);
    wasi_group = wasi(group_idx_mask);
    subject_indices_in_group = find(group_idx_mask);
    [~, sort_idx] = sort(wasi_group, 'ascend');
    
    bottom_10_idx = subject_indices_in_group(sort_idx(1:N_extremes));
    top_10_idx = subject_indices_in_group(sort_idx(end-N_extremes+1:end));
    
    % Calculate maps and time series for Bottom 10
    fprintf('  Calculating for Bottom 10 %s...\n', current_group.Name);
    [bottom_Q1_map_z, bottom_Q4_map_z, bottom_Q1_ts, bottom_Q4_ts, bottom_smooth_ts, bottom_smooth_ts_ind] = ...
        calculate_quartile_data(bottom_10_idx, all_elects, ciPLV_global, Nroi, q1_indices, q4_indices, total_timepoints);
    
    % Calculate maps and time series for Top 10
    fprintf('  Calculating for Top 10 %s...\n', current_group.Name);
    [top_Q1_map_z, top_Q4_map_z, top_Q1_ts, top_Q4_ts, top_smooth_ts, top_smooth_ts_ind] = ...
        calculate_quartile_data(top_10_idx, all_elects, ciPLV_global, Nroi, q1_indices, q4_indices, total_timepoints);
    
    all_q1_topomaps{g, 1} = bottom_Q1_map_z; 
    all_q1_topomaps{g, 2} = top_Q1_map_z;
    all_q4_topomaps{g, 1} = bottom_Q4_map_z;
    all_q4_topomaps{g, 2} = top_Q4_map_z;
    
    all_q1_timeseries{g, 1} = bottom_Q1_ts;
    all_q1_timeseries{g, 2} = top_Q1_ts;
    all_q4_timeseries{g, 1} = bottom_Q4_ts;
    all_q4_timeseries{g, 2} = top_Q4_ts;
    
    all_smoothness_ts_avg{g, 1} = bottom_smooth_ts;
    all_smoothness_ts_avg{g, 2} = top_smooth_ts;
    
    all_smoothness_ts_ind{g, 1} = bottom_smooth_ts_ind;
    all_smoothness_ts_ind{g, 2} = top_smooth_ts_ind;
end

%% --- 4. Determine Global Colormaps & Y-Limits ---
% Find the single, shared colormap for all Q1 topoplots
q1_max_val = 0;
for i = 1:numel(all_q1_topomaps), q1_max_val = max(q1_max_val, max(abs(all_q1_topomaps{i}))); end
if q1_max_val == 0, q1_max_val = 1; end; q1_clim = [-q1_max_val, q1_max_val];

% Find the single, shared colormap for all Q4 topoplots
q4_max_val = 0;
for i = 1:numel(all_q4_topomaps), q4_max_val = max(q4_max_val, max(abs(all_q4_topomaps{i}))); end
if q4_max_val == 0, q4_max_val = 1; end; q4_clim = [-q4_max_val, q4_max_val];

% Find the single, shared Y-axis limit for all Q1/Q4 time series plots
global_ts_min = inf; global_ts_max = -inf;
for i = 1:numel(all_q1_timeseries)
    global_ts_min = min([global_ts_min, min(all_q1_timeseries{i}), min(all_q4_timeseries{i})]);
    global_ts_max = max([global_ts_max, max(all_q1_timeseries{i}), max(all_q4_timeseries{i})]);
end
y_padding = (global_ts_max - global_ts_min) * 0.1;
ts_ylim = [global_ts_min - y_padding, global_ts_max + y_padding];


%% --- 5. Plot Single Figure (4x3 Layout) ---
disp('Plotting combined energy figure...');
fig = figure('Name', 'Spatial Distribution and Time Evolution of GSP Energy in WASI Extremes', 'WindowState', 'maximized');
t = tiledlayout(4, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- CONTROLS ---
nexttile; topoplot(all_q1_topomaps{1, 2}, EEG.chanlocs, 'maplimits', q1_clim); colorbar; title('Controls - Top 10 WASI - Q1 (Coupled) Topo'); ylabel('Controls');
nexttile; plot(time_vec_sec, all_q1_timeseries{1, 2}, 'b'); hold on; plot(time_vec_sec, all_q4_timeseries{1, 2}, 'r'); hold off; title('Controls - Top 10 WASI - Energy Evolution'); ylabel('Avg. Energy'); legend({'Q1', 'Q4'}, 'FontSize', 7); set(gca, 'YLim', ts_ylim, 'XLim', [time_vec_sec(1), time_vec_sec(end)]); 
nexttile; topoplot(all_q4_topomaps{1, 2}, EEG.chanlocs, 'maplimits', q4_clim); colorbar; title('Controls - Top 10 WASI - Q4 (Decoupled) Topo');
nexttile; topoplot(all_q1_topomaps{1, 1}, EEG.chanlocs, 'maplimits', q1_clim); colorbar; title('Controls - Bottom 10 WASI - Q1 (Coupled) Topo');
nexttile; plot(time_vec_sec, all_q1_timeseries{1, 1}, 'b'); hold on; plot(time_vec_sec, all_q4_timeseries{1, 1}, 'r'); hold off; title('Controls - Bottom 10 WASI - Energy Evolution'); ylabel('Avg. Energy'); set(gca, 'YLim', ts_ylim, 'XLim', [time_vec_sec(1), time_vec_sec(end)]); 
nexttile; topoplot(all_q4_topomaps{1, 1}, EEG.chanlocs, 'maplimits', q4_clim); colorbar; title('Controls - Bottom 10 WASI - Q4 (Decoupled) Topo');

% --- PATIENTS ---
nexttile; topoplot(all_q1_topomaps{2, 2}, EEG.chanlocs, 'maplimits', q1_clim); colorbar; title('Patients - Top 10 WASI - Q1 (Coupled) Topo'); ylabel('Patients');
nexttile; plot(time_vec_sec, all_q1_timeseries{2, 2}, 'b'); hold on; plot(time_vec_sec, all_q4_timeseries{2, 2}, 'r'); hold off; title('Patients - Top 10 WASI - Energy Evolution'); ylabel('Avg. Energy'); set(gca, 'YLim', ts_ylim, 'XLim', [time_vec_sec(1), time_vec_sec(end)]); 
nexttile; topoplot(all_q4_topomaps{2, 2}, EEG.chanlocs, 'maplimits', q4_clim); colorbar; title('Patients - Top 10 WASI - Q4 (Decoupled) Topo');
nexttile; topoplot(all_q1_topomaps{2, 1}, EEG.chanlocs, 'maplimits', q1_clim); colorbar; title('Patients - Bottom 10 WASI - Q1 (Coupled) Topo'); xlabel('Q1 Topoplots');
nexttile; plot(time_vec_sec, all_q1_timeseries{2, 1}, 'b'); hold on; plot(time_vec_sec, all_q4_timeseries{2, 1}, 'r'); hold off; title('Patients - Bottom 10 WASI - Energy Evolution'); ylabel('Avg. Energy'); xlabel('Time (seconds)'); set(gca, 'YLim', ts_ylim, 'XLim', [time_vec_sec(1), time_vec_sec(end)]); 
nexttile; topoplot(all_q4_topomaps{2, 1}, EEG.chanlocs, 'maplimits', q4_clim); colorbar; title('Patients - Bottom 10 WASI - Q4 (Decoupled) Topo'); xlabel('Q4 Topoplots');

sgtitle(t, 'Spatial Distribution and Time Evolution of GSP Energy in WASI Extremes (Avg. Z-scored Maps)', 'FontSize', 16, 'FontWeight', 'bold');

%% --- 6. Plot Smoothness Evolution Figure ---
disp('Plotting smoothness evolution figure...');
fig_smooth = figure('Name', 'Smoothness Evolution for WASI Extremes', 'WindowState', 'maximized');
t_smooth = tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

% Plot Controls
ax_c = nexttile;
plot(ax_c, time_vec_sec, all_smoothness_ts_avg{1, 2}, 'b', 'LineWidth', 1.5); % Top 10
hold(ax_c, 'on');
plot(ax_c, time_vec_sec, all_smoothness_ts_avg{1, 1}, 'r', 'LineWidth', 1.5); % Bottom 10
hold(ax_c, 'off');
title(ax_c, 'Controls: Average Instantaneous Smoothness');
ylabel(ax_c, 'Smoothness (s^T L s)');
legend(ax_c, {'Top 10 WASI', 'Bottom 10 WASI'}, 'Location', 'northwest');
set(ax_c, 'XLim', [time_vec_sec(1), time_vec_sec(end)]);

% Plot Patients
ax_p = nexttile;
plot(ax_p, time_vec_sec, all_smoothness_ts_avg{2, 2}, 'b', 'LineWidth', 1.5); % Top 10
hold(ax_p, 'on');
plot(ax_p, time_vec_sec, all_smoothness_ts_avg{2, 1}, 'r', 'LineWidth', 1.5); % Bottom 10
hold(ax_p, 'off');
title(ax_p, 'Patients: Average Instantaneous Smoothness');
ylabel(ax_p, 'Smoothness (s^T L s)');
xlabel(ax_p, 'Time (seconds)');
legend(ax_p, {'Top 10 WASI', 'Bottom 10 WASI'}, 'Location', 'northwest');
set(ax_p, 'XLim', [time_vec_sec(1), time_vec_sec(end)]);
set(ax_p, 'YLim', [0, 6000]);

sgtitle(t_smooth, 'Average Instantaneous Smoothness for WASI Extremes', 'FontSize', 16, 'FontWeight', 'bold');

%% --- 7. (NEW) Plot Histograms of Absolute Spike Amplitude ---
disp('Plotting histograms of absolute smoothness spike magnitude...');
fig_hist = figure('Name', 'Histogram of Smoothness Spike Magnitudes for WASI Extremes', 'WindowState', 'maximized');
t_hist = tiledlayout(2, 1, 'Padding', 'compact', 'TileSpacing', 'compact');

% --- Process Controls ---
ax_hist_c = nexttile;
ts_top_c = all_smoothness_ts_ind{1, 2}; 
ts_bot_c = all_smoothness_ts_ind{1, 1}; 
baseline_data_c = ts_bot_c(:); 
threshold_c = mean(baseline_data_c, 'omitnan') + spike_threshold_sd * std(baseline_data_c, 0, 'omitnan');
spikes_top_c = ts_top_c(ts_top_c > threshold_c);
spikes_bot_c = ts_bot_c(ts_bot_c > threshold_c);
[p_c, ~] = ranksum(spikes_top_c, spikes_bot_c);
set(ax_hist_c, 'XLim', [0, 6000]);

% --- CORRECTED KERNEL DENSITY PLOTTING ---
hold(ax_hist_c, 'on');

% --- Check and plot 'Top 10' ---
if ~isempty(spikes_top_c)
    [f_top_c, x_top_c] = ksdensity(spikes_top_c, 'Support', [min(spikes_top_c)*0.99, max(spikes_top_c)*1.01]);
    % Added 'r' (for red) as the color argument
    fill(ax_hist_c, x_top_c, f_top_c, 'b', 'FaceAlpha', 0.7, 'DisplayName', 'Top 10 WASI (Controls)', 'EdgeColor', 'none');
end

% --- Check and plot 'Bottom 10' ---
if ~isempty(spikes_bot_c)
    [f_bot_c, x_bot_c] = ksdensity(spikes_bot_c, 'Support', [min(spikes_bot_c)*0.99, max(spikes_bot_c)*1.01]);
    % Added 'b' (for blue) as the color argument
    fill(ax_hist_c, x_bot_c, f_bot_c, 'r', 'FaceAlpha', 0.7, 'DisplayName', 'Bottom 10 WASI (Controls)', 'EdgeColor', 'none');
end

xline(threshold_c,'k','LineStyle','--', 'LineWidth', 2.5, 'DisplayName', 'Threshold')

hold(ax_hist_c, 'off');
% --- END CORRECTION ---
if p>=0.0001
    title(ax_hist_c, sprintf('Controls: Distribution of Spike Magnitudes (p = %.4f)', p_c));
else
    title(ax_hist_c, 'Controls: Distribution of Spike Magnitudes (p < 0.0001)');
end
xlabel(ax_hist_c, 'Amplitude of Spikes (s^T L s)');
ylabel(ax_hist_c, 'Probability Density');
legend(ax_hist_c, 'Location', 'best'); % Added 'Location' for better placement

% --- Process Patients ---
ax_hist_p = nexttile;
ts_top_p = all_smoothness_ts_ind{2, 2}; 
ts_bot_p = all_smoothness_ts_ind{2, 1};
baseline_data_p = ts_bot_p(:); 
threshold_p = mean(baseline_data_p, 'omitnan') + spike_threshold_sd * std(baseline_data_p, 0, 'omitnan');
spikes_top_p = ts_top_p(ts_top_p > threshold_p);
spikes_bot_p = ts_bot_p(ts_bot_p > threshold_p);
[p_p, ~] = ranksum(spikes_top_p, spikes_bot_p);
set(ax_hist_p, 'XLim', [0, 6000]);


% --- CORRECTED KERNEL DENSITY PLOTTING ---
hold(ax_hist_p, 'on');

% --- Check and plot 'Top 10' ---
if ~isempty(spikes_top_p)
    [f_top_p, x_top_p] = ksdensity(spikes_top_p,'Support', [min(spikes_top_p)*0.99, max(spikes_top_p)*1.01]);
    % Added 'r' (for red) as the color argument
    fill(ax_hist_p, x_top_p, f_top_p, 'b', 'FaceAlpha', 0.7, 'DisplayName', 'Top 10 WASI (Patients)', 'EdgeColor', 'none');
end

% --- Check and plot 'Bottom 10' ---
if ~isempty(spikes_bot_p)
    [f_bot_p, x_bot_p] = ksdensity(spikes_bot_p, 'Support', [min(spikes_bot_p)*0.99, max(spikes_bot_p)*1.01]);
    % Added 'b' (for blue) as the color argument
    fill(ax_hist_p, x_bot_p, f_bot_p, 'r', 'FaceAlpha', 0.7, 'DisplayName', 'Bottom 10 WASI (Patients)', 'EdgeColor', 'none');
end

xline(threshold_p,'k','LineStyle','--', 'LineWidth', 2.5, 'DisplayName', 'Threshold')


hold(ax_hist_p, 'off');
% --- END CORRECTION ---
if p>=0.0001
    title(ax_hist_p, sprintf('Patients: Distribution of Spike Magnitudes (p = %.4f)', p_p));
else
    title(ax_hist_p, 'Patients: Distribution of Spike Magnitudes (p < 0.0001)');
end
xlabel(ax_hist_p, 'Amplitude of Spikes (s^T L s)');
ylabel(ax_hist_p, 'Probability Density');
legend(ax_hist_p, 'Location', 'best');

sgtitle(t_hist, sprintf('Distribution of Smoothness Spike Magnitudes (Threshold = Mean + %.1f SD of Bottom 10)', spike_threshold_sd), 'FontSize', 16, 'FontWeight', 'bold');

disp('Plotting complete.');


%% --- **MODIFIED** Helper Function: Calculate-then-Average Quartile Maps & Time Series ---
function [avg_zmap_Q1, avg_zmap_Q4, avg_ts_Q1, avg_ts_Q4, avg_smoothness_ts, all_smoothness_ts] = ...
    calculate_quartile_data(subject_indices, all_elects_4D, all_ciPLV, Nroi, q1_indices, q4_indices, total_timepoints)
    
    n_group = length(subject_indices);
    all_zmaps_Q1 = nan(Nroi, n_group);
    all_zmaps_Q4 = nan(Nroi, n_group);
    all_ts_Q1 = nan(n_group, total_timepoints);
    all_ts_Q4 = nan(n_group, total_timepoints);
    all_smoothness_ts = nan(n_group, total_timepoints); 
    
    % Use a standard 'for' loop for stability
    for i = 1:n_group
        subj_idx = subject_indices(i);
        
        % 1. Get this subject's graph
        A = squeeze(all_ciPLV(:,:,subj_idx));
        if all(A(:)==0) || any(isnan(A(:)))
            continue; 
        end
        
        D = diag(sum(A, 2));
        L = eye(Nroi) - diag(1./sqrt(diag(D) + eps)) * A * diag(1./sqrt(diag(D) + eps));
        if ~issymmetric(L), L = (L + L') / 2; end
        
        [V_subj, L_subj_diag] = eig(L);
        [eigenvals_sorted, idx] = sort(diag(L_subj_diag)); 
        V_subj = V_subj(:, idx);
    
        % 2. Get this subject's signal
        signal_subj_ts = reshape(squeeze(all_elects_4D(:,:,:,subj_idx)), Nroi, []); 
        signal_subj_power = real(signal_subj_ts).^2;
    
        % 3. GFT
        gft_coeffs = V_subj' * signal_subj_power; % [Nroi x timepoints]
        
        % 4. Filter for Q1
        gft_coeffs_Q1 = gft_coeffs;
        gft_coeffs_Q1(setdiff(1:Nroi, q1_indices), :) = 0;
        signal_Q1 = V_subj * gft_coeffs_Q1;
        
        % 5. Filter for Q4
        gft_coeffs_Q4 = gft_coeffs;
        gft_coeffs_Q4(setdiff(1:Nroi, q4_indices), :) = 0;
        signal_Q4 = V_subj * gft_coeffs_Q4;
        
        % 6. Calculate Nodal Energy Maps (sum across time)
        energy_map_Q1 = sum(real(signal_Q1).^2, 2);
        energy_map_Q4 = sum(real(signal_Q4).^2, 2);
        
        % 7. Z-score each map *within* the subject
        all_zmaps_Q1(:, i) = zscore(energy_map_Q1, 0, 1);
        all_zmaps_Q4(:, i) = zscore(energy_map_Q4, 0, 1);
        
        % 8. Calculate Energy Time Series (sum across nodes)
        all_ts_Q1(i, :) = sum(real(signal_Q1).^2, 1);
        all_ts_Q4(i, :) = sum(real(signal_Q4).^2, 1);
        
        % 9. Calculate Instantaneous Smoothness Time Series
        energy_t = abs(gft_coeffs).^2; % [Nroi x timepoints]
        all_smoothness_ts(i, :) = eigenvals_sorted' * energy_t;
    end
    
    % 10. Average the final *z-scored* maps across subjects
    avg_zmap_Q1 = mean(all_zmaps_Q1, 2, 'omitnan');
    avg_zmap_Q4 = mean(all_zmaps_Q4, 2, 'omitnan');
    
    % 11. Average the final *time series* across subjects
    avg_ts_Q1 = mean(all_ts_Q1, 1, 'omitnan');
    avg_ts_Q4 = mean(all_ts_Q4, 1, 'omitnan');
    
    % 12. Average the final *smoothness time series*
    avg_smoothness_ts = mean(all_smoothness_ts, 1, 'omitnan');
    % (all_smoothness_ts is also returned to be used for spike analysis)
end