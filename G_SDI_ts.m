%% SCRIPT 6: Structural Decoupling Index (SDI) Calculation - TIME SERIES SIGNAL
% This script implements the Preti & Van De Ville method using the
% reconstructed alpha time series ('all_elects') as the signal and each
% subject's global functional connectome ('ciPLV_global') as the structure.
% 1. Loads preprocessed time series and global graphs.
% 2. Calculates a single median split based on average global graph energy.
% 3. Loops through subjects:
%    a. Calculates graph harmonics for the subject's global graph.
%    b. Decomposes the subject's full alpha time series using these harmonics.
%    c. Calculates nodal SDI based on energy ratio across time.
% 4. Performs permutation testing (max-T/max-R) for group differences & WASI correlations.
% 5. Saves and visualizes all results.

clearvars -except all_elects;
clc;

%% --- 1. Load Preprocessed Data ---
disp('Loading preprocessed time series and graph data...');
%load('K:\JMECP_EEG_Analysis\results\RELAXv2\Preprocessed_GED_timeseries.mat');
load('K:\JMECP_EEG_Analysis\results\RELAXv2\Preprocessed_GED_Graphs_5s.mat'); % Load graphs and WASI
% Ensure variable consistency if names differ slightly
if exist('all_elects', 'var'), alpha_ts_4D = all_elects; clear all_elects; end
disp('Data loaded.');

%% --- 2. Prepare Time Series Signal Data ---
disp('Preparing time series data...');
% Reshape to nodes x subjects x total_timepoints
% Note: We do NOT z-score the time series itself for this analysis
alpha_ts_full = reshape(alpha_ts_4D, Nroi, nsubj, []);
clear alpha_ts_4D

%% --- 3. Calculate Average Energy Spectrum & Median Split (Using Global Graphs for Stability) ---
disp('Calculating average energy spectrum from global graphs to find median split...');
avg_energy_spectrum = zeros(Nroi, 1);
num_valid_spectra = 0;

for subi = 1:nsubj
    A = squeeze(ciPLV_global(:,:,subi));
    if all(A(:)==0) || any(isnan(A(:))), continue; end

    D = diag(sum(A, 2));
    L = eye(Nroi) - diag(1./sqrt(diag(D) + eps)) * A * diag(1./sqrt(diag(D) + eps));
    [V, Lambda] = eig(L);
    [~, idx] = sort(diag(Lambda));
    V_subj_global = V(:, idx);

    signal_subj = squeeze(alpha_ts_full(:, subi, :)); % nodes x timepoints
    gft_coeffs = V_subj_global' * signal_subj;
    energy_spectrum_subj = mean(abs(gft_coeffs).^2, 2);

    avg_energy_spectrum = avg_energy_spectrum + energy_spectrum_subj;
    num_valid_spectra = num_valid_spectra + 1;
end

avg_energy_spectrum = avg_energy_spectrum / num_valid_spectra;
avg_energy_spectrum = avg_energy_spectrum / sum(avg_energy_spectrum);

cumulative_energy = cumsum(avg_energy_spectrum);
median_split_idx = find(cumulative_energy >= 0.5, 1, 'first');
if isempty(median_split_idx), median_split_idx = floor(Nroi/2); end
fprintf('Global Median energy split occurs at harmonic index C = %d\n', median_split_idx);

%% --- 4. Calculate Subject-Specific SDI using Dynamic (Trial) Graphs ---
disp('Calculating subject-specific SDI maps using dynamic trial graphs...');
sdi_log2_avg = nan(Nroi, nsubj); % Final average SDI per subject
samples_per_trial = ltrials*EEG.srate;

% --- Start Parallel Pool ---
if isempty(gcp('nocreate')), parpool; end
disp('Parallel pool started for SDI calculation.');

parfor subi = 1:nsubj
    fprintf('Processing SDI for subject %d of %d\n', subi, nsubj);

    sdi_log2_trials_subj = nan(Nroi, ntrials); % Store SDI for each trial of this subject

    for triali = 1:ntrials
        % **MODIFIED**: Get trial-specific graph
        A_trial = squeeze(WC_TDA(:,:,triali, subi));
        if all(A_trial(:)==0) || any(isnan(A_trial(:))) || ~isreal(A_trial)
            %fprintf('  Skipping trial %d for subject %d due to invalid graph.\n', triali, subi);
            continue;
        end

        % Calculate trial-specific Laplacian and harmonics
        D_trial = diag(sum(A_trial, 2));
        L_trial = eye(Nroi) - diag(1./sqrt(diag(D_trial) + eps)) * A_trial * diag(1./sqrt(diag(D_trial) + eps));
        if ~issymmetric(L_trial), L_trial = (L_trial + L_trial') / 2; end

        [V_trial, Lambda_trial] = eig(L_trial);
        [~, idx_trial] = sort(diag(Lambda_trial));
        V_trial = V_trial(:, idx_trial);

        % **MODIFIED**: Get the time series segment for this specific trial
        start_idx = (triali - 1) * samples_per_trial + 1;
        end_idx = triali * samples_per_trial;
        signal_trial_segment = squeeze(alpha_ts_full(:, subi, start_idx:end_idx)); % nodes x samples_per_trial

        % Graph Fourier Transform (GFT) for this trial segment
        gft_coeffs_trial = V_trial' * signal_trial_segment; % Nroi x samples_per_trial

        % --- Filter Signal using the GLOBAL median_split_idx ---
        gft_coeffs_coupled = gft_coeffs_trial;
        gft_coeffs_coupled(median_split_idx+1:end, :) = 0;
        signal_coupled = V_trial * gft_coeffs_coupled; % nodes x samples_per_trial

        gft_coeffs_decoupled = gft_coeffs_trial;
        gft_coeffs_decoupled(1:median_split_idx, :) = 0;
        signal_decoupled = V_trial * gft_coeffs_decoupled; % nodes x samples_per_trial

        % --- Calculate Energy Ratio (SDI) per Node for this trial ---
        % Sum energy across time points within the trial segment
        energy_coupled = real(sum(signal_coupled.^2, 2));
        energy_decoupled = real(sum(signal_decoupled.^2, 2));

        energy_coupled(energy_coupled < eps) = eps;
        sdi_ratio = energy_decoupled ./ energy_coupled;
        sdi_log2_trials_subj(:, triali) = log2(max(eps, sdi_ratio));

    end % End trial loop

    % Average SDI across valid trials for this subject
    sdi_log2_avg(:, subi) = mean(sdi_log2_trials_subj, 2, 'omitnan');

end % End subject parfor loop

disp('SDI calculation complete.');
sdi_log2 = sdi_log2_avg;

%% --- 5. Statistical Analysis 1: Group Difference (Max-T Permutation) ---
disp('Performing permutation test with max-T correction for group difference...');
n_perms = 5000;
stat_alpha = 0.05;

data_controls = sdi_log2(:, origLabels == 0);
data_patients = sdi_log2(:, origLabels == 1);
data_all = [data_controls, data_patients];
labels_all = [zeros(1, size(data_controls, 2)), ones(1, size(data_patients, 2))];

t_real = nan(Nroi, 1);
for chan = 1:Nroi
    data_ctrl_chan = data_controls(chan, ~isnan(data_controls(chan,:)));
    data_pat_chan  = data_patients(chan, ~isnan(data_patients(chan,:)));
    if length(data_ctrl_chan) > 1 && length(data_pat_chan) > 1
        [~, ~, ~, stats] = ttest2(data_ctrl_chan, data_pat_chan);
        if isfield(stats, 'tstat'), t_real(chan) = stats.tstat; end
    end
end

max_t_stat_perm = nan(n_perms, 1);
disp('Running permutations for group difference...');
parfor i = 1:n_perms
    perm_labels = labels_all(randperm(length(labels_all)));
    t_perm = nan(Nroi, 1);
    for chan = 1:Nroi
        data_perm_ctrl = data_all(chan, perm_labels == 0);
        data_perm_pat = data_all(chan, perm_labels == 1);
        data_perm_ctrl = data_perm_ctrl(~isnan(data_perm_ctrl));
        data_perm_pat = data_perm_pat(~isnan(data_perm_pat));
        if length(data_perm_ctrl) > 1 && length(data_perm_pat) > 1
             [~, ~, ~, stats] = ttest2(data_perm_ctrl, data_perm_pat);
             if isfield(stats, 'tstat'), t_perm(chan) = stats.tstat; end
        end
    end
    max_t_val = max(abs(t_perm), [], 'omitnan');
    if ~isempty(max_t_val), max_t_stat_perm(i) = max_t_val; else, max_t_stat_perm(i) = 0; end
end
disp('Permutations complete.');

t_threshold = prctile(max_t_stat_perm, 100 * (1 - stat_alpha));
significant_electrodes_groupdiff = find(abs(t_real) >= t_threshold);
sig_mask_groupdiff = zeros(Nroi, 1);
if ~isempty(significant_electrodes_groupdiff)
    sig_mask_groupdiff(significant_electrodes_groupdiff) = 1;
    disp(['Found ' num2str(length(significant_electrodes_groupdiff)) ' significant electrode(s) for group difference.']);
else
    disp('No significant group differences found.');
end


%% --- 6. Statistical Analysis 2: WASI Correlation (Max-R Permutation) ---
disp('Performing permutation test with max-R correction for WASI correlation...');
groups_to_analyze = {struct('Name', 'Controls', 'LabelIndex', 0), struct('Name', 'Patients', 'LabelIndex', 1)};
corr_results = struct(); % Store results for both groups

for g = 1:length(groups_to_analyze)
    current_group = groups_to_analyze{g};
    disp(['Running correlations for: ' current_group.Name]);

    group_mask = (origLabels == current_group.LabelIndex);
    sdi_group = sdi_log2(:, group_mask);
    wasi_group = wasi(group_mask);
    n_group = size(sdi_group, 2);

    r_real = nan(Nroi, 1);
    for chan = 1:Nroi
        valid_mask = ~isnan(sdi_group(chan, :))' & ~isnan(wasi_group);
        if sum(valid_mask) > 2, r_real(chan) = corr(sdi_group(chan, valid_mask)', wasi_group(valid_mask)); end
    end

    max_r_stat_perm = nan(n_perms, 1);
    fprintf('Running permutations for %s...\n', current_group.Name);
    parfor i = 1:n_perms
        perm_wasi = wasi_group(randperm(n_group));
        r_perm = nan(Nroi, 1);
        for chan = 1:Nroi
            valid_mask = ~isnan(sdi_group(chan, :))' & ~isnan(perm_wasi);
            if sum(valid_mask) > 2, r_perm(chan) = corr(sdi_group(chan, valid_mask)', perm_wasi(valid_mask)); end
        end
        max_r_val = max(abs(r_perm), [], 'omitnan');
        if ~isempty(max_r_val), max_r_stat_perm(i) = max_r_val; else, max_r_stat_perm(i) = 0; end
    end
    disp('Permutations complete.');

    r_threshold = prctile(max_r_stat_perm, 100 * (1 - stat_alpha));
    significant_electrodes_corr = find(abs(r_real) >= r_threshold);
    sig_mask_corr = zeros(Nroi, 1);
    if ~isempty(significant_electrodes_corr)
        sig_mask_corr(significant_electrodes_corr) = 1;
        disp(['Found ' num2str(length(significant_electrodes_corr)) ' significant electrode(s) for WASI correlation in ' current_group.Name '.']);
    else
        disp(['No significant WASI correlations found in ' current_group.Name '.']);
    end

    corr_results.(current_group.Name).r_real = r_real;
    corr_results.(current_group.Name).r_threshold = r_threshold;
    corr_results.(current_group.Name).sig_mask = sig_mask_corr;
    corr_results.(current_group.Name).max_r_stat_perm = max_r_stat_perm;
end

%% --- 7. Statistical Analysis 3: Difference in WASI Correlations (Max-Diff Permutation) ---
disp('Performing permutation test on the difference between correlation maps...');

r_controls = corr_results.Controls.r_real;
r_patients = corr_results.Patients.r_real;
diff_map_real = r_patients - r_controls;

max_diff_stat_perm = nan(n_perms, 1);
disp('Running permutations for correlation differences...');
parfor i = 1:n_perms
    perm_labels = labels_all(randperm(length(labels_all)));
    r_perm_group0 = nan(Nroi, 1);
    r_perm_group1 = nan(Nroi, 1);
    for chan = 1:Nroi
        perm_mask0 = (perm_labels == 0);
        valid_mask0 = ~isnan(sdi_log2(chan, perm_mask0))' & ~isnan(wasi(perm_mask0));
        if sum(valid_mask0) > 2, r_perm_group0(chan) = corr(sdi_log2(chan, perm_mask0(valid_mask0))', wasi(perm_mask0(valid_mask0))); end
        perm_mask1 = (perm_labels == 1);
        valid_mask1 = ~isnan(sdi_log2(chan, perm_mask1))' & ~isnan(wasi(perm_mask1));
        if sum(valid_mask1) > 2, r_perm_group1(chan) = corr(sdi_log2(chan, perm_mask1(valid_mask1))', wasi(perm_mask1(valid_mask1))); end
    end
    diff_map_perm = r_perm_group1 - r_perm_group0;
    max_diff_val = max(abs(diff_map_perm), [], 'omitnan');
    if ~isempty(max_diff_val), max_diff_stat_perm(i) = max_diff_val; else, max_diff_stat_perm(i) = 0; end
end
disp('Permutations complete.');

diff_threshold = prctile(max_diff_stat_perm, 100 * (1 - stat_alpha));
significant_electrodes_corr_diff = find(abs(diff_map_real) >= diff_threshold);
sig_mask_corr_diff = zeros(Nroi, 1);
if ~isempty(significant_electrodes_corr_diff)
    sig_mask_corr_diff(significant_electrodes_corr_diff) = 1;
    disp(['Found ' num2str(length(significant_electrodes_corr_diff)) ' electrode(s) with significantly different WASI correlations between groups.']);
else
    disp('No significant differences found in WASI correlation patterns between groups.');
end

corr_results.Difference.diff_map_real = diff_map_real;
corr_results.Difference.diff_threshold = diff_threshold;
corr_results.Difference.sig_mask = sig_mask_corr_diff;
corr_results.Difference.max_diff_stat_perm = max_diff_stat_perm;


%% --- 8. Save and Visualize Results ---
disp('Saving SDI results...');
save('K:\JMECP_EEG_Analysis\results\RELAXv2\SDI_Results_TimeSeries_DynamicGraphs.mat', ...
    'sdi_log2', 'median_split_idx', 'origLabels', 'wasi', 'EEG', 'Nroi', 'nsubj', 'ntrials', ...
    'significant_electrodes_groupdiff', 'sig_mask_groupdiff', 't_real', 't_threshold', 'max_t_stat_perm', ...
    'corr_results');

disp('Plotting average SDI maps and significant differences...');
sdi_controls_avg = mean(sdi_log2(:, origLabels == 0), 2, 'omitnan');
sdi_patients_avg = mean(sdi_log2(:, origLabels == 1), 2, 'omitnan');
sdi_diff_avg = sdi_patients_avg - sdi_controls_avg;

max_abs_sdi_avg = max(abs([sdi_controls_avg(:); sdi_patients_avg(:)]));
common_maplimits_avg = [-max_abs_sdi_avg, max_abs_sdi_avg];

fig_avg = figure('Name', 'Average SDI & Significant Differences (Time Series Signal, Dynamic Graphs)');
t_avg = tiledlayout(2, 2, 'Padding', 'compact');

nexttile(t_avg); topoplot(sdi_controls_avg, EEG.chanlocs, 'maplimits', common_maplimits_avg); colorbar; title('Controls Avg SDI (log2)');
nexttile(t_avg); topoplot(sdi_patients_avg, EEG.chanlocs, 'maplimits', common_maplimits_avg); colorbar; title('Patients Avg SDI (log2)');
nexttile(t_avg); topoplot(sdi_diff_avg, EEG.chanlocs, 'maplimits', 'absmax'); colorbar; title('Difference (Patients - Controls)');
nexttile(t_avg); topoplot(sdi_diff_avg.*sig_mask_groupdiff, EEG.chanlocs, 'maplimits', 'absmax', 'style', 'map', 'emarker2', {find(sig_mask_groupdiff),'o','k',5,1}); colorbar; title(['Significant Differences (p < ' num2str(stat_alpha) ' maxT-corr.)']);

ax_main_title_avg = axes('Parent', fig_avg, 'Position', [0 0 1 1], 'Visible', 'off');
title_text_avg = 'Structural Decoupling Index (Alpha Time Series on Dynamic Trial FC Graphs)';
text(0.5, 0.97, title_text_avg, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold', 'Parent', ax_main_title_avg);

% Plot WASI correlations within groups
for g = 1:length(groups_to_analyze)
    current_group = groups_to_analyze{g};
    fig_corr = figure('Name', ['SDI-WASI Correlation (Time Series, Dynamic Graphs) - ' current_group.Name]);
    t_corr = tiledlayout(1, 2, 'Padding', 'compact');
    nexttile(t_corr); r_map = corr_results.(current_group.Name).r_real; topoplot(r_map, EEG.chanlocs, 'maplimits', 'absmax'); colorbar; title(['Correlation with WASI (r-values) - ' current_group.Name]);
    nexttile(t_corr); sig_mask_corr = corr_results.(current_group.Name).sig_mask; topoplot(r_map.*sig_mask_corr, EEG.chanlocs, 'maplimits', 'absmax', 'style', 'map', 'emarker2', {find(sig_mask_corr),'o','k',5,1}); colorbar; title(['Significant Correlations (p < ' num2str(stat_alpha) ' maxR-corr.)']);
    ax_main_title_corr = axes('Parent', fig_corr, 'Position', [0 0 1 1], 'Visible', 'off'); title_text_corr = ['SDI (Time Series, Dyn Graphs) Correlation with WASI - ' current_group.Name]; text(0.5, 0.97, title_text_corr, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold', 'Parent', ax_main_title_corr);
end

% Plot the difference in correlations between groups
fig_corr_diff = figure('Name', 'Difference in SDI-WASI Correlation (Time Series, Dynamic Graphs)');
t_corr_diff = tiledlayout(1, 2, 'Padding', 'compact');
nexttile(t_corr_diff); diff_map = corr_results.Difference.diff_map_real; topoplot(diff_map, EEG.chanlocs, 'maplimits', 'absmax'); colorbar; title('Difference Map (r_{patients} - r_{controls})');
nexttile(t_corr_diff); sig_mask_diff = corr_results.Difference.sig_mask; topoplot(diff_map.*sig_mask_diff, EEG.chanlocs, 'maplimits', 'absmax', 'style', 'map', 'emarker2', {find(sig_mask_diff),'o','k',5,1}); colorbar; title(['Significant Differences (p < ' num2str(stat_alpha) ' maxDiff-corr.)']);
ax_main_title_diff = axes('Parent', fig_corr_diff, 'Position', [0 0 1 1], 'Visible', 'off'); title_text_diff = 'Difference in SDI (Time Series, Dyn Graphs)-WASI Correlation Between Groups'; text(0.5, 0.97, title_text_diff, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 14, 'FontWeight', 'bold', 'Parent', ax_main_title_diff);

disp('SDI analysis finished.');

%% --- Helper Functions ---
function chan_hood = spatial_neighbors(chanlocs)
    if isempty(chanlocs) || ~isfield(chanlocs, 'X') || ~isfield(chanlocs, 'Y'), error('Valid chanlocs structure needed.'); end
    coords = [[chanlocs.X]', [chanlocs.Y]'];
    valid_coords_idx = ~isnan(coords(:,1)) & ~isnan(coords(:,2));
    coords = coords(valid_coords_idx, :);
    original_indices = find(valid_coords_idx);
    if size(coords,1) < 3, warning('Not enough valid coords.'); chan_hood = zeros(length(chanlocs)); return; end
    dt = delaunayTriangulation(coords); adj = dt.edges;
    N = length(chanlocs); chan_hood = zeros(N, N);
    for i = 1:size(adj, 1)
        orig_idx1 = original_indices(adj(i,1)); orig_idx2 = original_indices(adj(i,2));
        chan_hood(orig_idx1, orig_idx2) = 1; chan_hood(orig_idx2, orig_idx1) = 1;
    end
end

function clusters = find_clusters(t_values, p_values, chan_hood, alpha)
    clusters = struct('channels', {}, 'stat', {});
    t_values = t_values(:); p_values = p_values(:);
    sig_chans = find(p_values < alpha);
    if isempty(sig_chans), return; end
    visited = false(size(t_values)); cluster_idx = 0;
    for chan_idx = 1:length(sig_chans)
        chan = sig_chans(chan_idx);
        if visited(chan), continue; end
        cluster_idx = cluster_idx + 1; current_cluster_chans = []; queue = chan; visited(chan) = true;
        while ~isempty(queue)
            current_chan = queue(1); queue(1) = []; current_cluster_chans(end+1) = current_chan;
            neighbor_indices = find(chan_hood(current_chan, :));
            valid_neighbors = neighbor_indices(neighbor_indices <= length(p_values));
            significant_unvisited_neighbors = valid_neighbors(p_values(valid_neighbors) < alpha & ~visited(valid_neighbors));
            for neighbor = significant_unvisited_neighbors
                visited(neighbor) = true; queue(end+1) = neighbor;
            end
        end
        clusters(cluster_idx).channels = current_cluster_chans;
        clusters(cluster_idx).stat = sum(abs(t_values(current_cluster_chans)));
    end
end
