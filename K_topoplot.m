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

%% --- 3. Calculate All Energy Maps and Time Series First ---
disp('Calculating all energy maps and time series for all groups and extremes...');

all_q1_topomaps = cell(2, 2); % (Group, Extreme)
all_q4_topomaps = cell(2, 2);
all_q1_timeseries = cell(2, 2);
all_q4_timeseries = cell(2, 2);

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
    [bottom_Q1_map_z, bottom_Q4_map_z, bottom_Q1_ts, bottom_Q4_ts] = ...
        calculate_quartile_data(bottom_10_idx, all_elects, ciPLV_global, Nroi, q1_indices, q4_indices, total_timepoints);
    
    % Calculate maps and time series for Top 10
    fprintf('  Calculating for Top 10 %s...\n', current_group.Name);
    [top_Q1_map_z, top_Q4_map_z, top_Q1_ts, top_Q4_ts] = ...
        calculate_quartile_data(top_10_idx, all_elects, ciPLV_global, Nroi, q1_indices, q4_indices, total_timepoints);
    
    % Store the final z-scored maps
    all_q1_topomaps{g, 1} = bottom_Q1_map_z; % Row 1=Controls, Row 2=Patients; Col 1=Bottom 10
    all_q1_topomaps{g, 2} = top_Q1_map_z;    % Col 2=Top 10
    all_q4_topomaps{g, 1} = bottom_Q4_map_z;
    all_q4_topomaps{g, 2} = top_Q4_map_z;
    
    % Store the final average time series
    all_q1_timeseries{g, 1} = bottom_Q1_ts;
    all_q1_timeseries{g, 2} = top_Q1_ts;
    all_q4_timeseries{g, 1} = bottom_Q4_ts;
    all_q4_timeseries{g, 2} = top_Q4_ts;
end

%% --- 4. Determine Global Colormaps & Y-Limits ---
% Find the single, shared colormap for all Q1 topoplots
q1_max_val = 0;
for i = 1:numel(all_q1_topomaps)
    q1_max_val = max(q1_max_val, max(abs(all_q1_topomaps{i})));
end
if q1_max_val == 0, q1_max_val = 1; end % Failsafe
q1_clim = [-q1_max_val, q1_max_val];

% Find the single, shared colormap for all Q4 topoplots
q4_max_val = 0;
for i = 1:numel(all_q4_topomaps)
    q4_max_val = max(q4_max_val, max(abs(all_q4_topomaps{i})));
end
if q4_max_val == 0, q4_max_val = 1; end % Failsafe
q4_clim = [-q4_max_val, q4_max_val];

% **NEW**: Find the single, shared Y-axis limit for all time series plots
global_ts_min = inf;
global_ts_max = -inf;
for i = 1:numel(all_q1_timeseries)
    global_ts_min = min(global_ts_min, min(all_q1_timeseries{i}));
    global_ts_max = max(global_ts_max, max(all_q1_timeseries{i}));
    global_ts_min = min(global_ts_min, min(all_q4_timeseries{i}));
    global_ts_max = max(global_ts_max, max(all_q4_timeseries{i}));
end
y_padding = (global_ts_max - global_ts_min) * 0.1;
ts_ylim = [global_ts_min - y_padding, global_ts_max + y_padding];


%% --- 5. Plot Single Figure ---
disp('Plotting combined figure...');
fig = figure('Name', 'Spatial Distribution and Time Evolution of GSP Energy in WASI Extremes', 'WindowState', 'maximized');
t = tiledlayout(4, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- CONTROLS ---
% Row 1: Controls - Top 10
nexttile; % Col 1
topoplot(all_q1_topomaps{1, 2}, EEG.chanlocs, 'maplimits', q1_clim); colorbar;
title('Controls - Top 10 WASI - Q1 (Coupled) Topo');
ylabel('Controls');

nexttile; % Col 2
plot(time_vec_sec, all_q1_timeseries{1, 2}, 'b', 'LineWidth', 1); hold on;
plot(time_vec_sec, all_q4_timeseries{1, 2}, 'r', 'LineWidth', 1); hold off;
title('Controls - Top 10 WASI - Energy Evolution'); % **FIX**: Title updated
ylabel('Avg. Energy'); legend({'Q1 (Coupled)', 'Q4 (Decoupled)'}, 'FontSize', 7);
set(gca, 'YLim', ts_ylim, 'XLim', [time_vec_sec(1), time_vec_sec(end)]); % **FIX**: Set YLim

nexttile; % Col 3
topoplot(all_q4_topomaps{1, 2}, EEG.chanlocs, 'maplimits', q4_clim); colorbar;
title('Controls - Top 10 WASI - Q4 (Decoupled) Topo');

% Row 2: Controls - Bottom 10
nexttile; % Col 1
topoplot(all_q1_topomaps{1, 1}, EEG.chanlocs, 'maplimits', q1_clim); colorbar;
title('Controls - Bottom 10 WASI - Q1 (Coupled) Topo');

nexttile; % Col 2
plot(time_vec_sec, all_q1_timeseries{1, 1}, 'b', 'LineWidth', 1); hold on;
plot(time_vec_sec, all_q4_timeseries{1, 1}, 'r', 'LineWidth', 1); hold off;
title('Controls - Bottom 10 WASI - Energy Evolution'); % **FIX**: Title updated
ylabel('Avg. Energy');
set(gca, 'YLim', ts_ylim, 'XLim', [time_vec_sec(1), time_vec_sec(end)]); % **FIX**: Set YLim

nexttile; % Col 3
topoplot(all_q4_topomaps{1, 1}, EEG.chanlocs, 'maplimits', q4_clim); colorbar;
title('Controls - Bottom 10 WASI - Q4 (Decoupled) Topo');

% --- PATIENTS ---
% Row 3: Patients - Top 10
nexttile; % Col 1
topoplot(all_q1_topomaps{2, 2}, EEG.chanlocs, 'maplimits', q1_clim); colorbar;
title('Patients - Top 10 WASI - Q1 (Coupled) Topo');
ylabel('Patients');

nexttile; % Col 2
plot(time_vec_sec, all_q1_timeseries{2, 2}, 'b', 'LineWidth', 1); hold on;
plot(time_vec_sec, all_q4_timeseries{2, 2}, 'r', 'LineWidth', 1); hold off;
title('Patients - Top 10 WASI - Energy Evolution'); % **FIX**: Title updated
ylabel('Avg. Energy');
set(gca, 'YLim', ts_ylim, 'XLim', [time_vec_sec(1), time_vec_sec(end)]); % **FIX**: Set YLim

nexttile; % Col 3
topoplot(all_q4_topomaps{2, 2}, EEG.chanlocs, 'maplimits', q4_clim); colorbar;
title('Patients - Top 10 WASI - Q4 (Decoupled) Topo');

% Row 4: Patients - Bottom 10
nexttile; % Col 1
topoplot(all_q1_topomaps{2, 1}, EEG.chanlocs, 'maplimits', q1_clim); colorbar;
title('Patients - Bottom 10 WASI - Q1 (Coupled) Topo');
xlabel('Q1 Topoplots');

nexttile; % Col 2
plot(time_vec_sec, all_q1_timeseries{2, 1}, 'b', 'LineWidth', 1); hold on;
plot(time_vec_sec, all_q4_timeseries{2, 1}, 'r', 'LineWidth', 1); hold off;
title('Patients - Bottom 10 WASI - Energy Evolution'); % **FIX**: Title updated
ylabel('Avg. Energy'); xlabel('Time (seconds)'); % **FIX**: X-label updated
set(gca, 'YLim', ts_ylim, 'XLim', [time_vec_sec(1), time_vec_sec(end)]); % **FIX**: Set YLim

nexttile; % Col 3
topoplot(all_q4_topomaps{2, 1}, EEG.chanlocs, 'maplimits', q4_clim); colorbar;
title('Patients - Bottom 10 WASI - Q4 (Decoupled) Topo');
xlabel('Q4 Topoplots');

% Add a single, robust main title
sgtitle(t, 'Spatial Distribution and Time Evolution of GSP Energy in WASI Extremes (Avg. Z-scored Maps)', 'FontSize', 16, 'FontWeight', 'bold');

disp('Plotting complete.');


%% --- **MODIFIED** Helper Function: Calculate-then-Average Quartile Maps & Time Series ---
function [avg_zmap_Q1, avg_zmap_Q4, avg_ts_Q1, avg_ts_Q4] = ...
    calculate_quartile_data(subject_indices, all_elects_4D, all_ciPLV, Nroi, q1_indices, q4_indices, total_timepoints)
    
    n_group = length(subject_indices);
    all_zmaps_Q1 = nan(Nroi, n_group);
    all_zmaps_Q4 = nan(Nroi, n_group);
    all_ts_Q1 = nan(n_group, total_timepoints);
    all_ts_Q4 = nan(n_group, total_timepoints);
    
    % **FIX**: Removed parfor, using standard for loop
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
        [~, idx] = sort(diag(L_subj_diag));
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
    end
    
    % 9. Average the final *z-scored* maps across subjects
    avg_zmap_Q1 = mean(all_zmaps_Q1, 2, 'omitnan');
    avg_zmap_Q4 = mean(all_zmaps_Q4, 2, 'omitnan');
    
    % 10. Average the final *time series* across subjects
    avg_ts_Q1 = mean(all_ts_Q1, 1, 'omitnan');
    avg_ts_Q4 = mean(all_ts_Q4, 1, 'omitnan');
end