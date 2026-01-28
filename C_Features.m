%% SCRIPT 2: Feature Calculation, Statistical Analysis, and Plotting
% This script loads preprocessed data and performs a focused feature analysis.
% 1. Loads preprocessed data and extracts WASI scores.
% 2. Calculates topological features (node strength).
% 3. Calculates a focused set of GSP features: smoothness, spectral entropy, and frequency spread.
% 4. Calculates dynamic features (std, mafd, SampEn) for the GSP time series.
% 5. Performs statistical tests for group differences and WASI correlations.
% 6. Generates summary figures for all FDR-corrected significant results.

%clear all;
clc;

%% --- 1. Load Preprocessed Data ---
disp('Loading preprocessed data...');
%load('K:\JMECP_EEG_Analysis\results\RELAXv2\RELAXv2_iPLV_EC_180s_5strials_8to12hz_252elec_142subj_shrunk_vfinal.mat');
disp('Data loaded.');

%% --- 2. Extract Behavioral Scores ---
disp('Extracting and matching WASI, QOLIE, and CBCL scores...');
spreadsheet_path = 'K:\JMECP_EEG_Analysis\05-29-24 JMECP FULL DATA EXTRACT (plus Psych Data).xlsx';
wasi_table = readtable(spreadsheet_path, 'Range','IR1:IW141');
qolie_table = readtable(spreadsheet_path, 'Range', 'UY1:UY141');
cbcl_table = readtable(spreadsheet_path, 'Range', 'OY1:OY141');

behavCode = str2double(table2array(readtable(spreadsheet_path, 'Range','A1:A141')));

% Use the subject codes from the loaded .mat file to ensure correct matching
[memb_orig, newidx] = ismember(behavCode, subcode);
newidx = newidx(memb_orig);

% Reorder WASI scores
wasi_table_sorted = wasi_table(memb_orig,:);
wasi_table_sorted = wasi_table_sorted(newidx,:);
wasi = table2array(wasi_table_sorted(:,5));

% **FIX**: Reorder and convert QOLIE from string/empty cells to numeric/NaN
qolie_table_sorted = qolie_table(memb_orig,:);
qolie_table_sorted = qolie_table_sorted(newidx,:);
% Use str2double, which correctly handles strings and empty cells -> NaN
qolie = str2double(table2cell(qolie_table_sorted));

% **FIX**: Reorder and convert CBCL from string/empty cells to numeric/NaN
cbcl_table_sorted = cbcl_table(memb_orig,:);
cbcl_table_sorted = cbcl_table_sorted(newidx,:);
cbcl = str2double(table2cell(cbcl_table_sorted));

% Now we repeat for g factor
spreadsheet_path2 = 'K:\JMECP_EEG_Analysis\JME_07222024_psych_cog_clusters_g_psych_clusters_hdEEGvars.xlsx';    
behavCode = str2double(table2array(readtable(spreadsheet_path2, 'Range','B1:B141')));
[memb_orig, newidx] = ismember(behavCode, subcode);
newidx = newidx(memb_orig);

g_table = readtable(spreadsheet_path2, 'Range','WH1:WH141');
g_table_sorted = g_table(memb_orig,:);
g_table_sorted = g_table_sorted(newidx,:);
g_factor = table2array(g_table_sorted);

%% --- 3. Calculate Topological Features (Node Strength) ---
disp('Calculating topological node strength features...');
WC_TDA = abs(WC_TDA);
ciPLV_global = abs(ciPLV_global);

all_nodeavg0 = nan(Nroi, nsubj, ntrials);
all_nodeavg1 = nan(Nroi, nsubj, ntrials);

for triali = 1:ntrials
    [allWb, allWd] = WS_decompose_batch(squeeze(WC_TDA(:,:,triali,:)));
    all_nodeavg0(:,:,triali) = calculate_node_strength(allWb, Nroi);
    all_nodeavg1(:,:,triali) = calculate_node_strength(allWd, Nroi);
end

[allWb_global, allWd_global] = WS_decompose_batch(ciPLV_global);
nodeavg0_global = calculate_node_strength(allWb_global, Nroi);
nodeavg1_global = calculate_node_strength(allWd_global, Nroi);

data_0z = zscore(all_nodeavg0, 0, 1);
data_1z = zscore(all_nodeavg1, 0, 1);
data0z_global = zscore(nodeavg0_global, 0, 1);
data1z_global = zscore(nodeavg1_global, 0, 1);

%% --- 4. Calculate Graph Signal Processing (GSP) Features ---
disp('Calculating focused trial-based GSP features...');
featureNames_GSP = {'smoothness', 'spectral_entropy', 'freq_spread'};

smoothness_0D_trials = nan(ntrials, nsubj);
spectral_entropy_0D_trials = nan(ntrials, nsubj);
freq_spread_0D_trials = nan(ntrials, nsubj);
smoothness_1D_trials = nan(ntrials, nsubj);
spectral_entropy_1D_trials = nan(ntrials, nsubj);
freq_spread_1D_trials = nan(ntrials, nsubj);

for triali = 1:ntrials
    fprintf('  Processing GSP for Trial %d of %d\n', triali, ntrials);
    parfor subi = 1:nsubj
        A = squeeze(WC_TDA(:,:,triali,subi));
        D = diag(sum(A, 2));
        L = eye(size(A)) - diag(1./sqrt(diag(D) + eps)) * A * diag(1./sqrt(diag(D) + eps));
        [V, Lambda] = eig(L);
        [eigenvals, idx] = sort(diag(Lambda));
        V = V(:, idx);

        signal_0D = data_0z(:, subi, triali);
        signal_1D = data_1z(:, subi, triali);

        gsp_results_0D = calculate_gsp_features(signal_0D, L, V, eigenvals, featureNames_GSP);
        gsp_results_1D = calculate_gsp_features(signal_1D, L, V, eigenvals, featureNames_GSP);
        
        smoothness_0D_trials(triali, subi) = gsp_results_0D.smoothness;
        spectral_entropy_0D_trials(triali, subi) = gsp_results_0D.spectral_entropy;
        freq_spread_0D_trials(triali, subi) = gsp_results_0D.freq_spread;
        
        smoothness_1D_trials(triali, subi) = gsp_results_1D.smoothness;
        spectral_entropy_1D_trials(triali, subi) = gsp_results_1D.spectral_entropy;
        freq_spread_1D_trials(triali, subi) = gsp_results_1D.freq_spread;
    end
end

features_0D.smoothness = smoothness_0D_trials;
features_0D.spectral_entropy = spectral_entropy_0D_trials;
features_0D.freq_spread = freq_spread_0D_trials;
features_1D.smoothness = smoothness_1D_trials;
features_1D.spectral_entropy = spectral_entropy_1D_trials;
features_1D.freq_spread = freq_spread_1D_trials;


disp('Calculating focused global GSP features...');
smoothness_0D_global = nan(1, nsubj);
spectral_entropy_0D_global = nan(1, nsubj);
freq_spread_0D_global = nan(1, nsubj);
smoothness_1D_global = nan(1, nsubj);
spectral_entropy_1D_global = nan(1, nsubj);
freq_spread_1D_global = nan(1, nsubj);

parfor subi = 1:nsubj
    A = squeeze(ciPLV_global(:,:,subi));
    D = diag(sum(A, 2));
    L = eye(size(A)) - diag(1./sqrt(diag(D) + eps)) * A * diag(1./sqrt(diag(D) + eps));
    [V, Lambda] = eig(L);
    [eigenvals, idx] = sort(diag(Lambda));
    V = V(:, idx);

    signal_0D = data0z_global(:, subi);
    signal_1D = data1z_global(:, subi);

    gsp_results_0D = calculate_gsp_features(signal_0D, L, V, eigenvals, featureNames_GSP);
    gsp_results_1D = calculate_gsp_features(signal_1D, L, V, eigenvals, featureNames_GSP);
    
    smoothness_0D_global(subi) = gsp_results_0D.smoothness;
    spectral_entropy_0D_global(subi) = gsp_results_0D.spectral_entropy;
    freq_spread_0D_global(subi) = gsp_results_0D.freq_spread;
    
    smoothness_1D_global(subi) = gsp_results_1D.smoothness;
    spectral_entropy_1D_global(subi) = gsp_results_1D.spectral_entropy;
    freq_spread_1D_global(subi) = gsp_results_1D.freq_spread;
end

features_0Dglobal.smoothness = smoothness_0D_global;
features_0Dglobal.spectral_entropy = spectral_entropy_0D_global;
features_0Dglobal.freq_spread = freq_spread_0D_global;
features_1Dglobal.smoothness = smoothness_1D_global;
features_1Dglobal.spectral_entropy = spectral_entropy_1D_global;
features_1Dglobal.freq_spread = freq_spread_1D_global;

%% --- 5. Calculate Dynamic Features (Std, MAFD, SampEn) ---
disp('Calculating dynamic features from GSP time series...');
base_feature_names = fieldnames(features_0D);
dynamic_features_0D = struct();
dynamic_features_1D = struct();

for i = 1:length(base_feature_names)
    fname = base_feature_names{i};
    
    ts_0D = features_0D.(fname);
    dynamic_features_0D.([fname '_std']) = std(ts_0D, 0, 1, 'omitnan');
    dynamic_features_0D.([fname '_mafd']) = mean(abs(diff(ts_0D)), 1, 'omitnan');
    all_subject_stds_0D = std(ts_0D, 0, 1, 'omitnan');
    global_r_0D = 0.2 * median(all_subject_stds_0D, 'omitnan');
    sampen_0D = nan(1, nsubj);
    for subi = 1:nsubj
        sampen_0D(subi) = sampen(ts_0D(:, subi), 2, global_r_0D);
    end
    dynamic_features_0D.([fname '_sampen']) = sampen_0D;

    ts_1D = features_1D.(fname);
    dynamic_features_1D.([fname '_std']) = std(ts_1D, 0, 1, 'omitnan');
    dynamic_features_1D.([fname '_mafd']) = mean(abs(diff(ts_1D)), 1, 'omitnan');
    all_subject_stds_1D = std(ts_1D, 0, 1, 'omitnan');
    global_r_1D = 0.2 * median(all_subject_stds_1D, 'omitnan');
    sampen_1D = nan(1, nsubj);
    for subi = 1:nsubj
        sampen_1D(subi) = sampen(ts_1D(:, subi), 2, global_r_1D);
    end
    dynamic_features_1D.([fname '_sampen']) = sampen_1D;
end
disp('All features calculated.');

%% --- 6. Statistical Analyses and Plotting ---

n_perms = 5000; % Number of permutations for all tests

% --- 6A: Group Difference Analysis ---
disp('Starting combined group difference analysis with permutation testing...');
data_sources = {features_0D, features_1D, features_0Dglobal, features_1Dglobal};
source_names = {'0D Trial-Averaged', '1D Trial-Averaged', '0D Global', '1D Global'};
all_significant_results_groupdiff = cell(4, 1);
groupLabels = {'Controls', 'Patients'};
featureNames = fieldnames(features_0D);

for s = 1:length(data_sources)
    num_features = length(featureNames);
    feature_matrix = nan(nsubj, num_features);
    for f = 1:num_features
        feature_data = data_sources{s}.(featureNames{f});
        if contains(source_names{s}, 'Trial')
            feature_matrix(:, f) = mean(feature_data, 1, 'omitnan');
        else
            feature_matrix(:, f) = feature_data(:);
        end
    end
    
    all_significant_results_groupdiff{s} = run_permutation_test_ttest2(feature_matrix, origLabels, featureNames, n_perms);
end
plot_group_differences(all_significant_results_groupdiff, source_names, groupLabels, origLabels);
disp('Group difference analysis complete.');

% --- 6B: WASI Correlation Analysis (Static Features) ---
disp('Starting WASI correlation analysis for static features with permutation testing...');
groups_to_analyze = {struct('Name', 'Controls', 'LabelIndex', 0), struct('Name', 'Patients', 'LabelIndex', 1)};
all_data.trial_0D = features_0D; all_data.trial_1D = features_1D;
all_data.global_0D = features_0Dglobal; all_data.global_1D = features_1Dglobal;
data_types = {'trial_0D', 'trial_1D', 'global_0D', 'global_1D'};
data_type_names = {'0D Trial-Averaged', '1D Trial-Averaged', '0D Global', '1D Global'};

for g = 1:length(groups_to_analyze)
    current_group = groups_to_analyze{g};
    all_significant_results_wasi = cell(4, 1);
    for s = 1:length(data_types)
        num_features = length(featureNames);
        feature_matrix = nan(nsubj, num_features);
        for f = 1:num_features
            feature_data = all_data.(data_types{s}).(featureNames{f});
            if contains(data_types{s}, 'trial')
                feature_matrix(:, f) = mean(feature_data, 1, 'omitnan');
            else
                feature_matrix(:, f) = feature_data(:);
            end
        end
        all_significant_results_wasi{s} = run_permutation_test_corr(feature_matrix, wasi, origLabels, current_group.LabelIndex, featureNames, n_perms);
    end
    plot_wasi_correlations(all_significant_results_wasi, [current_group.Name ' Only'], data_type_names);
end
disp('Static WASI correlation analysis complete.');

% --- 6C: WASI Correlation Analysis (Dynamic Features) ---
disp('Starting WASI correlation analysis for dynamic features with permutation testing...');
dynamic_sources = {dynamic_features_0D, dynamic_features_1D};
dynamic_source_names = {'Dynamics of 0D Features', 'Dynamics of 1D Features'};

for g = 1:length(groups_to_analyze)
    current_group = groups_to_analyze{g};
    all_significant_results_dyn = cell(2, 1);
    for s = 1:length(dynamic_sources)
        featureNames_dyn = fieldnames(dynamic_sources{s});
        num_features = length(featureNames_dyn);
        feature_matrix = nan(nsubj, num_features);
        for f = 1:num_features
             feature_matrix(:,f) = dynamic_sources{s}.(featureNames_dyn{f})(:);
        end
        all_significant_results_dyn{s} = run_permutation_test_corr(feature_matrix, wasi, origLabels, current_group.LabelIndex, featureNames_dyn, n_perms);
    end
    plot_dynamic_wasi_correlations(all_significant_results_dyn, [current_group.Name ' Only'], dynamic_source_names);
end
disp('Dynamic WASI correlation analysis complete.');


%% --- Helper Functions ---
function significant_results = run_permutation_test_ttest2(feature_matrix, group_labels, feature_names, n_perms)
    % Performs permutation testing for group differences using max-T correction.
    
    group0_idx = find(group_labels == 0);
    group1_idx = find(group_labels == 1);
    
    % 1. Calculate the real t-statistics
    real_t_stats = nan(1, size(feature_matrix, 2));
    for f = 1:size(feature_matrix, 2)
        [~, ~, ~, stats] = ttest2(feature_matrix(group0_idx, f), feature_matrix(group1_idx, f));
        if isfield(stats, 'tstat'), real_t_stats(f) = stats.tstat; end
    end
    
    % 2. Build the null distribution of the maximum t-statistic
    max_t_dist = nan(n_perms, 1);
    all_labels = group_labels;
    for i = 1:n_perms
        perm_labels = all_labels(randperm(length(all_labels)));
        perm_group0_idx = find(perm_labels == 0);
        perm_group1_idx = find(perm_labels == 1);
        
        perm_t_stats = nan(1, size(feature_matrix, 2));
        for f = 1:size(feature_matrix, 2)
            [~, ~, ~, stats] = ttest2(feature_matrix(perm_group0_idx, f), feature_matrix(perm_group1_idx, f));
            if isfield(stats, 'tstat'), perm_t_stats(f) = stats.tstat; end
        end
        max_t_dist(i) = max(abs(perm_t_stats), [], 'omitnan');
    end
    
    % 3. Find the significance threshold
    t_threshold = prctile(max_t_dist, 95);
    
    % 4. Identify significant results
    significant_indices = find(abs(real_t_stats) >= t_threshold);
    significant_results = {};
    for idx = significant_indices
        res.name = feature_names{idx};
        res.q_value = 1 - (find(sort(max_t_dist) <= abs(real_t_stats(idx)), 1, 'last') / n_perms); % Approximate p-value
        res.data = feature_matrix(:, idx)';
        significant_results{end+1} = res;
    end
    
    if ~isempty(significant_results)
        q_vals_for_sorting = cellfun(@(x) x.q_value, significant_results);
        [~, sort_idx] = sort(q_vals_for_sorting);
        significant_results = significant_results(sort_idx);
    end
end

function significant_results = run_permutation_test_corr(feature_matrix, score_vector, group_labels, group_idx, feature_names, n_perms)
    % Performs permutation testing for correlations using max-correlation correction.
    
    group_mask = (group_labels == group_idx);
    
    % 1. Calculate the real correlations
    real_r_stats = nan(1, size(feature_matrix, 2));
    valid_group_indices = find(group_mask);
    
    for f = 1:size(feature_matrix, 2)
        valid_data_mask = ~isnan(feature_matrix(valid_group_indices, f)) & ~isnan(score_vector(valid_group_indices));
        if sum(valid_data_mask) > 2
             real_r_stats(f) = corr(feature_matrix(valid_group_indices(valid_data_mask), f), score_vector(valid_group_indices(valid_data_mask)));
        end
    end

    % 2. Build the null distribution of the maximum correlation
    max_r_dist = nan(n_perms, 1);
    group_scores = score_vector(group_mask);
    group_features = feature_matrix(group_mask, :);
    
    for i = 1:n_perms
        perm_scores = group_scores(randperm(length(group_scores)));
        perm_r_stats = nan(1, size(group_features, 2));
        for f = 1:size(group_features, 2)
            valid_data_mask = ~isnan(group_features(:, f)) & ~isnan(perm_scores);
             if sum(valid_data_mask) > 2
                perm_r_stats(f) = corr(group_features(valid_data_mask, f), perm_scores(valid_data_mask));
             end
        end
        max_r_dist(i) = max(abs(perm_r_stats), [], 'omitnan');
    end
    
    % 3. Find the significance threshold
    r_threshold = prctile(max_r_dist, 95);

    % 4. Identify significant results
    significant_indices = find(abs(real_r_stats) >= r_threshold);
    significant_results = {};
    for idx = significant_indices
        res.name = feature_names{idx};
        res.r = real_r_stats(idx);
        res.q_value = 1 - (find(sort(max_r_dist) <= abs(res.r), 1, 'last') / n_perms);
        
        % Get data for plotting
        valid_plot_mask = ~isnan(feature_matrix(valid_group_indices, idx)) & ~isnan(score_vector(valid_group_indices));
        res.feature_data = feature_matrix(valid_group_indices(valid_plot_mask), idx);
        res.wasi_data = score_vector(valid_group_indices(valid_plot_mask));
        significant_results{end+1} = res;
    end
    
    if ~isempty(significant_results)
        q_vals_for_sorting = cellfun(@(x) x.q_value, significant_results);
        [~, sort_idx] = sort(q_vals_for_sorting);
        significant_results = significant_results(sort_idx);
    end
end


function [Wb, Wd] = WS_decompose_batch(W)
    n_subjects = size(W, 3); p = size(W, 1);
    Wb = nan(p-1, 3, n_subjects);
    num_death_edges = (p * (p-1) / 2) - (p - 1);
    Wd = nan(num_death_edges, 3, n_subjects);
    for i = 1:n_subjects
        Wi = W(:,:,i); [rows, cols] = find(triu(ones(p), 1));
        weights = Wi(sub2ind(size(Wi), rows, cols)); G1 = graph(rows, cols, weights, p);
        birthMtx1 = conncomp_birth(Wi); Wb(:,:,i) = birthMtx1;
        death_edges = rmedge(G1, birthMtx1(:,1), birthMtx1(:,2)).Edges{:,:};
        if size(death_edges,1) == num_death_edges, Wd(:,:,i) = death_edges; end
    end
end

function birthMtx = conncomp_birth(adj)
    g = graph(-adj, 'upper', 'omitselfloops'); gTree = minspantree(g);
    birthMtx = gTree.Edges{:,:}; birthMtx(:,3) = birthMtx(:,3) * -1;
end

function node_strength = calculate_node_strength(all_W, Nroi)
    num_subjects = size(all_W, 3); node_strength = zeros(Nroi, num_subjects);
    for subi = 1:num_subjects
        W = squeeze(all_W(:,:,subi));
        if isempty(W) || any(isnan(W(:))), continue; end
        node_strength(:, subi) = accumarray([W(:,1); W(:,2)], [W(:,3); W(:,3)], [Nroi 1]);
    end
end

function results = calculate_gsp_features(signal, L, V, eigenvals, feature_names)
    results = struct();
    gft_coeffs = V' * signal; energy = gft_coeffs.^2;
    total_energy = sum(energy); if total_energy < eps, total_energy = 1; end
    energy_norm = energy / total_energy;
    
    if ismember('smoothness', feature_names), results.smoothness = signal' * L * signal; end
    if ismember('spectral_entropy', feature_names), results.spectral_entropy = -sum(energy_norm .* log(energy_norm + eps)); end
    if ismember('freq_spread', feature_names)
        mean_freq = sum(eigenvals .* energy) / total_energy;
        variance = sum(eigenvals.^2 .* energy) / total_energy - mean_freq^2;
        results.freq_spread = sqrt(max(0, variance));
    end
end

function se = sampen(y, m, r)
    y = y(~isnan(y));
    y = y(:)'; n = length(y);
    if n < m + 1 || (r+eps) <= 0 || isnan(r), se = NaN; return; end
    r = r + eps;
    
    N_m = n - m; count_m = 0;
    for i = 1:N_m, for j = i+1:N_m
        if max(abs(y(i:i+m-1) - y(j:j+m-1))) < r, count_m = count_m + 1; end
    end, end
    N_m1 = n - m - 1; count_m1 = 0;
    for i = 1:N_m1, for j = i+1:N_m1
        if max(abs(y(i:i+m) - y(j:j+m))) < r, count_m1 = count_m1 + 1; end
    end, end
    if count_m == 0 || count_m1 == 0, se = NaN; else, se = -log(count_m1 / count_m); end
end

function plot_group_differences(all_significant_results, source_names, groupLabels, origLabels_in)
    fig = figure('Name', 'Group Difference Summary', 'WindowState', 'maximized');
    panel_positions = {[0.05 0.53 0.43 0.38], [0.52 0.53 0.43 0.38], [0.05 0.08 0.43 0.38], [0.52 0.08 0.43 0.38]};
    for s = 1:4
        pnl = uipanel('Parent', fig, 'Title', source_names{s}, 'FontSize', 14, 'Position', panel_positions{s});
        if ~isempty(all_significant_results{s})
            t_nested = tiledlayout(pnl, 'flow', 'TileSpacing', 'compact', 'Padding', 'compact');
            for p = 1:length(all_significant_results{s})
                res = all_significant_results{s}{p}; nexttile(t_nested);
                boxplot(res.data, origLabels_in, 'Labels', groupLabels);
                title(sprintf('%s\n(p \\approx %.3f)', strrep(res.name, '_', ' '), res.q_value), 'FontSize', 9);
                ylabel('Feature Value'); grid on;
            end
        else
            ax_dummy = axes('Parent', pnl, 'Visible', 'off');
            text(0.5, 0.5, 'No Significant Results', 'Parent', ax_dummy, 'HorizontalAlignment', 'center', 'FontSize', 14);
        end
    end
    ax_main_title = axes('Parent', fig, 'Position', [0 0 1 1], 'Visible', 'off');
    title_text = 'Group Differences (Controls vs. Patients) - Permutation Corrected';
    text(0.5, 0.97, title_text, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 16, 'FontWeight', 'bold', 'Parent', ax_main_title);
end

function plot_wasi_correlations(all_significant_results, group_name, data_type_names)
    fig = figure('Name', ['WASI Correlations (Static) - ' group_name], 'WindowState', 'maximized');
    panel_positions = {[0.05 0.53 0.43 0.38], [0.52 0.53 0.43 0.38], [0.05 0.08 0.43 0.38], [0.52 0.08 0.43 0.38]};
    for s = 1:4
        pnl = uipanel('Parent', fig, 'Title', data_type_names{s}, 'FontSize', 14, 'Position', panel_positions{s});
        if ~isempty(all_significant_results{s})
            t_nested = tiledlayout(pnl, 'flow', 'TileSpacing', 'compact', 'Padding', 'compact');
            for p = 1:length(all_significant_results{s})
                res = all_significant_results{s}{p}; nexttile(t_nested);
                scatter(res.feature_data, res.wasi_data, 75, 'filled', 'MarkerFaceAlpha', 0.7); hold on;
                p_fit = polyfit(res.feature_data, res.wasi_data, 1);
                plot(linspace(min(res.feature_data), max(res.feature_data), 100), polyval(p_fit, linspace(min(res.feature_data), max(res.feature_data), 100)), 'r-', 'LineWidth', 2);
                title(sprintf('%s\nr=%.3f (p \\approx %.3f)', strrep(res.name, '_', ' '), res.r, res.q_value), 'FontSize', 9);
                xlabel('Feature Value'); ylabel('WASI Score'); grid on; hold off;
            end
        else
            ax_dummy = axes('Parent', pnl, 'Visible', 'off');
            text(0.5, 0.5, 'No Significant Results', 'Parent', ax_dummy, 'HorizontalAlignment', 'center', 'FontSize', 14);
        end
    end
    ax_main_title = axes('Parent', fig, 'Position', [0 0 1 1], 'Visible', 'off');
    title_text = ['Significant Correlations with WASI (Static) for ' group_name ' - Permutation Corrected'];
    text(0.5, 0.97, title_text, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 16, 'FontWeight', 'bold', 'Parent', ax_main_title);
end

function plot_dynamic_wasi_correlations(all_significant_results, group_name, data_type_names)
    fig = figure('Name', ['WASI Correlations (Dynamics) - ' group_name], 'WindowState', 'maximized');
    panel_positions = {[0.05 0.1 0.43 0.75], [0.52 0.1 0.43 0.75]};
    for s = 1:2
        pnl = uipanel('Parent', fig, 'Title', data_type_names{s}, 'FontSize', 14, 'Position', panel_positions{s});
        if ~isempty(all_significant_results{s})
            t_nested = tiledlayout(pnl, 'flow', 'TileSpacing', 'compact', 'Padding', 'compact');
            for p = 1:length(all_significant_results{s})
                res = all_significant_results{s}{p}; nexttile(t_nested);
                scatter(res.feature_data, res.wasi_data, 75, 'filled', 'MarkerFaceAlpha', 0.7); hold on;
                p_fit = polyfit(res.feature_data, res.wasi_data, 1);
                plot(linspace(min(res.feature_data), max(res.feature_data), 100), polyval(p_fit, linspace(min(res.feature_data), max(res.feature_data), 100)), 'r-', 'LineWidth', 2);
                title(sprintf('%s\nr=%.3f (p \\approx %.3f)', strrep(res.name, '_', ' '), res.r, res.q_value), 'FontSize', 9);
                xlabel('Dynamic Feature Value'); ylabel('WASI Score'); grid on; hold off;
            end
        else
            ax_dummy = axes('Parent', pnl, 'Visible', 'off');
            text(0.5, 0.5, 'No Significant Results', 'Parent', ax_dummy, 'HorizontalAlignment', 'center', 'FontSize', 14);
        end
    end
    ax_main_title = axes('Parent', fig, 'Position', [0 0 1 1], 'Visible', 'off');
    title_text = ['Significant Correlations with WASI (Dynamics) for ' group_name ' - Permutation Corrected'];
    text(0.5, 0.97, title_text, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 16, 'FontWeight', 'bold', 'Parent', ax_main_title);
end

