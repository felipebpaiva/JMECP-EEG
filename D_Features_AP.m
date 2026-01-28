%% SCRIPT 3: Alpha Power Feature Calculation, Analysis, and Plotting
% This script is analogous to Script 2, but uses the reconstructed alpha
% power as the signal for GSP analysis, instead of topological features.
% 1. Loads preprocessed data and checks/formats the alphaPower variable.
% 2. Calculates a focused set of GSP features from the alpha power signal.
% 3. Calculates dynamic features (std, mafd, SampEn) for the GSP time series.
% 4. Performs statistical tests using robust permutation testing.
% 5. Generates summary figures for all significant results.

%clear all;
clc;

%% --- 1. Load Preprocessed Data ---
disp('Loading preprocessed data...');
%load('K:\JMECP_EEG_Analysis\results\RELAXv2\Preprocessed_GED_Graphs.mat');
disp('Data loaded.');

%% --- 2. Prepare Alpha Power Data ---
disp('Preparing and Z-scoring alpha power data...');

% Check dimensions of alphaPower. Expected from Script 1 is (trials, nodes, subjects).
% The analysis scripts expect (nodes, subjects, trials).
if size(alphaPower, 1) == ntrials && size(alphaPower, 2) == Nroi
    disp('Permuting alphaPower to nodes x subjects x trials...');
    alphaPower_permuted = permute(alphaPower, [2, 3, 1]);
else
    alphaPower_permuted = alphaPower; % Assume it's already in the correct format
end

% Z-score alpha power across electrodes for each subject and trial
data_Az = zscore(alphaPower_permuted, 0, 1);

% Create global (trial-averaged) alpha power and z-score it
alphaPower_global = mean(alphaPower_permuted, 3);
dataAz_global = zscore(alphaPower_global, 0, 1);

%% --- Interlude: Inspect Alpha Maps ---
% disp('Inspecting 10 randomly-selected alpha maps...');
% 
% idx2plot = randi(nsubj,10,1);
% figure('Name', 'Random Subject Alpha Maps', 'WindowState', 'maximized');
% % **FIX**: Create the tiled layout *before* the loop
% t = tiledlayout(2, 5, 'TileSpacing', 'compact', 'Padding', 'compact');
% 
% for idx = 1:length(idx2plot)
%     subject_index = idx2plot(idx);
%     % **FIX**: Move to the next tile in the existing layout
%     nexttile(); 
%     topoplot(alphamap(subject_index,:), chanlocs);
%     colorbar;
%     title(['Subject Index: ' num2str(subject_index)]);
% end
% % Add a main title to the figure
% title(t, 'Average Alpha Component Maps for 10 Random Subjects', 'FontWeight', 'bold');

%% --- 3. Extract Behavioral Scores ---
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

%% --- 4. Calculate Alpha Power GSP Features ---
disp('Calculating trial-based GSP features for alpha power...');
featureNames_GSP = {'smoothness', 'spectral_entropy', 'freq_spread'};

% Initialize temporary matrices for parfor
smoothness_A_trials = nan(ntrials, nsubj);
spectral_entropy_A_trials = nan(ntrials, nsubj);
freq_spread_A_trials = nan(ntrials, nsubj);

for triali = 1:ntrials
    fprintf('  Processing GSP for Trial %d of %d\n', triali, ntrials);
    parfor subi = 1:nsubj
        A = squeeze(WC_TDA(:,:,triali,subi));
        D = diag(sum(A, 2));
        L = eye(size(A)) - diag(1./sqrt(diag(D) + eps)) * A * diag(1./sqrt(diag(D) + eps));
        [V, Lambda] = eig(L);
        [eigenvals, idx] = sort(diag(Lambda));
        V = V(:, idx);

        signal_A = data_Az(:, subi, triali);
        gsp_results_A = calculate_gsp_features(signal_A, L, V, eigenvals, featureNames_GSP);
        
        smoothness_A_trials(triali, subi) = gsp_results_A.smoothness;
        spectral_entropy_A_trials(triali, subi) = gsp_results_A.spectral_entropy;
        freq_spread_A_trials(triali, subi) = gsp_results_A.freq_spread;
    end
end

features_A.smoothness = smoothness_A_trials;
features_A.spectral_entropy = spectral_entropy_A_trials;
features_A.freq_spread = freq_spread_A_trials;

disp('Calculating global GSP features for alpha power...');
smoothness_Aglobal = nan(1, nsubj);
spectral_entropy_Aglobal = nan(1, nsubj);
freq_spread_Aglobal = nan(1, nsubj);

parfor subi = 1:nsubj
    A = squeeze(ciPLV_global(:,:,subi));
    D = diag(sum(A, 2));
    L = eye(size(A)) - diag(1./sqrt(diag(D) + eps)) * A * diag(1./sqrt(diag(D) + eps));
    [V, Lambda] = eig(L);
    [eigenvals, idx] = sort(diag(Lambda));
    V = V(:, idx);

    signal_Aglobal = dataAz_global(:, subi);
    gsp_results_Aglobal = calculate_gsp_features(signal_Aglobal, L, V, eigenvals, featureNames_GSP);
    
    smoothness_Aglobal(subi) = gsp_results_Aglobal.smoothness;
    spectral_entropy_Aglobal(subi) = gsp_results_Aglobal.spectral_entropy;
    freq_spread_Aglobal(subi) = gsp_results_Aglobal.freq_spread;
end

features_Aglobal.smoothness = smoothness_Aglobal;
features_Aglobal.spectral_entropy = spectral_entropy_Aglobal;
features_Aglobal.freq_spread = freq_spread_Aglobal;


%% --- 5. Calculate Dynamic Features for Alpha Power GSP ---
disp('Calculating dynamic features from Alpha Power GSP time series...');
base_feature_names_A = fieldnames(features_A);
dynamic_features_AP = struct();

for i = 1:length(base_feature_names_A)
    fname = base_feature_names_A{i};
    
    ts_A = features_A.(fname);
    dynamic_features_AP.([fname '_std']) = std(ts_A, 0, 1, 'omitnan');
    dynamic_features_AP.([fname '_mafd']) = mean(abs(diff(ts_A)), 1, 'omitnan');
    
    all_subject_stds_A = std(ts_A, 0, 1, 'omitnan');
    global_r_A = 0.2 * median(all_subject_stds_A, 'omitnan');
    
    sampen_A = nan(1, nsubj);
    for subi = 1:nsubj
        sampen_A(subi) = sampen(ts_A(:, subi), 2, global_r_A);
    end
    dynamic_features_AP.([fname '_sampen']) = sampen_A;
end
disp('All alpha power features calculated.');


%% --- 6. Statistical Analyses and Plotting ---

n_perms = 5000;
groupLabels = {'Controls', 'Patients'};

% --- 6A: Group Difference Analysis ---
disp('Starting group difference analysis for alpha power features...');
data_sources_A = {features_A, features_Aglobal};
source_names_A = {'Alpha Power Trial-Averaged', 'Alpha Power Global'};
all_significant_results_groupdiff_A = cell(2, 1);
featureNames_A = fieldnames(features_A);

for s = 1:length(data_sources_A)
    feature_matrix = nan(nsubj, length(featureNames_A));
    for f = 1:length(featureNames_A)
        feature_data = data_sources_A{s}.(featureNames_A{f});
        if contains(source_names_A{s}, 'Trial')
            feature_matrix(:, f) = mean(feature_data, 1, 'omitnan');
        else
            feature_matrix(:, f) = feature_data(:);
        end
    end
    all_significant_results_groupdiff_A{s} = run_permutation_test_ttest2(feature_matrix, origLabels, featureNames_A, n_perms);
end
plot_group_differences_A(all_significant_results_groupdiff_A, source_names_A, groupLabels, origLabels);


% --- 6B: WASI Correlation Analysis (Static Features) ---
disp('Starting WASI correlation analysis for static alpha power features...');
groups_to_analyze = {struct('Name', 'Controls', 'LabelIndex', 0), struct('Name', 'Patients', 'LabelIndex', 1)};

for g = 1:length(groups_to_analyze)
    current_group = groups_to_analyze{g};
    all_significant_results_wasi_A = cell(2, 1);
    for s = 1:length(data_sources_A)
        feature_matrix = nan(nsubj, length(featureNames_A));
        for f = 1:length(featureNames_A)
            feature_data = data_sources_A{s}.(featureNames_A{f});
            if contains(source_names_A{s}, 'Trial')
                feature_matrix(:, f) = mean(feature_data, 1, 'omitnan');
            else
                feature_matrix(:, f) = feature_data(:);
            end
        end
        all_significant_results_wasi_A{s} = run_permutation_test_corr(feature_matrix, wasi, origLabels, current_group.LabelIndex, featureNames_A, n_perms);
    end
    plot_wasi_correlations_A(all_significant_results_wasi_A, [current_group.Name ' Only'], source_names_A);
end


% --- 6C: WASI Correlation Analysis (Dynamic Features) ---
disp('Starting WASI correlation analysis for dynamic alpha power features...');
for g = 1:length(groups_to_analyze)
    current_group = groups_to_analyze{g};
    featureNames_dyn_A = fieldnames(dynamic_features_AP);
    feature_matrix_dyn = nan(nsubj, length(featureNames_dyn_A));
    for f = 1:length(featureNames_dyn_A)
        feature_matrix_dyn(:,f) = dynamic_features_AP.(featureNames_dyn_A{f})(:);
    end
    significant_results_dyn_A = run_permutation_test_corr(feature_matrix_dyn, wasi, origLabels, current_group.LabelIndex, featureNames_dyn_A, n_perms);
    plot_dynamic_wasi_correlations_A({significant_results_dyn_A}, [current_group.Name ' Only'], {'Dynamics of Alpha Power Features'});
end

disp('All analyses complete.');


%% --- Helper Functions ---
% (Note: These are copies from Script 2 for standalone execution)

function significant_results = run_permutation_test_ttest2(feature_matrix, group_labels, feature_names, n_perms)
    group0_idx = find(group_labels == 0);
    group1_idx = find(group_labels == 1);
    real_t_stats = nan(1, size(feature_matrix, 2));
    for f = 1:size(feature_matrix, 2)
        [~, ~, ~, stats] = ttest2(feature_matrix(group0_idx, f), feature_matrix(group1_idx, f));
        if isfield(stats, 'tstat'), real_t_stats(f) = stats.tstat; end
    end
    
    max_t_dist = nan(n_perms, 1);
    for i = 1:n_perms
        perm_labels = group_labels(randperm(length(group_labels)));
        perm_t_stats = nan(1, size(feature_matrix, 2));
        for f = 1:size(feature_matrix, 2)
            [~, ~, ~, stats] = ttest2(feature_matrix(perm_labels == 0, f), feature_matrix(perm_labels == 1, f));
            if isfield(stats, 'tstat'), perm_t_stats(f) = stats.tstat; end
        end
        max_t_dist(i) = max(abs(perm_t_stats), [], 'omitnan');
    end
    
    t_threshold = prctile(max_t_dist, 95);
    significant_indices = find(abs(real_t_stats) >= t_threshold);
    significant_results = {};
    for idx = significant_indices
        res.name = feature_names{idx};
        res.q_value = 1 - (find(sort(max_t_dist) <= abs(real_t_stats(idx)), 1, 'last') / n_perms);
        res.data = feature_matrix(:, idx)';
        significant_results{end+1} = res;
    end
    if ~isempty(significant_results)
        [~, sort_idx] = sort(cellfun(@(x) x.q_value, significant_results));
        significant_results = significant_results(sort_idx);
    end
end

function significant_results = run_permutation_test_corr(feature_matrix, score_vector, group_labels, group_idx, feature_names, n_perms)
    group_mask = (group_labels == group_idx);
    real_r_stats = nan(1, size(feature_matrix, 2));
    valid_group_indices = find(group_mask);
    
    for f = 1:size(feature_matrix, 2)
        valid_data_mask = ~isnan(feature_matrix(valid_group_indices, f)) & ~isnan(score_vector(valid_group_indices));
        if sum(valid_data_mask) > 2, real_r_stats(f) = corr(feature_matrix(valid_group_indices(valid_data_mask), f), score_vector(valid_group_indices(valid_data_mask))); end
    end

    max_r_dist = nan(n_perms, 1);
    group_scores = score_vector(group_mask);
    group_features = feature_matrix(group_mask, :);
    
    for i = 1:n_perms
        perm_scores = group_scores(randperm(length(group_scores)));
        perm_r_stats = nan(1, size(group_features, 2));
        for f = 1:size(group_features, 2)
            valid_data_mask = ~isnan(group_features(:, f)) & ~isnan(perm_scores);
             if sum(valid_data_mask) > 2, perm_r_stats(f) = corr(group_features(valid_data_mask, f), perm_scores(valid_data_mask)); end
        end
        max_r_dist(i) = max(abs(perm_r_stats), [], 'omitnan');
    end
    
    r_threshold = prctile(max_r_dist, 95);
    significant_indices = find(abs(real_r_stats) >= r_threshold);
    significant_results = {};
    for idx = significant_indices
        res.name = feature_names{idx};
        res.r = real_r_stats(idx);
        res.q_value = 1 - (find(sort(max_r_dist) <= abs(res.r), 1, 'last') / n_perms);
        valid_plot_mask = ~isnan(feature_matrix(valid_group_indices, idx)) & ~isnan(score_vector(valid_group_indices));
        res.feature_data = feature_matrix(valid_group_indices(valid_plot_mask), idx);
        res.wasi_data = score_vector(valid_group_indices(valid_plot_mask));
        significant_results{end+1} = res;
    end
    if ~isempty(significant_results)
        [~, sort_idx] = sort(cellfun(@(x) x.q_value, significant_results));
        significant_results = significant_results(sort_idx);
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
    y = y(~isnan(y)); y = y(:)'; n = length(y);
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

function plot_group_differences_A(all_significant_results, source_names, groupLabels, origLabels_in)
    fig = figure('Name', 'Alpha Power Group Difference Summary', 'WindowState', 'maximized');
    panel_positions = {[0.05 0.1 0.43 0.75], [0.52 0.1 0.43 0.75]};
    for s = 1:length(source_names)
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
    title(ax_main_title, 'Alpha Power Group Differences - Permutation Corrected', 'FontSize', 16, 'FontWeight', 'bold');
end

function plot_wasi_correlations_A(all_significant_results, group_name, data_type_names)
    fig = figure('Name', ['Alpha Power WASI Correlations (Static) - ' group_name], 'WindowState', 'maximized');
    panel_positions = {[0.05 0.1 0.43 0.75], [0.52 0.1 0.43 0.75]};
    for s = 1:length(data_type_names)
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
    title(ax_main_title, ['Alpha Power Static Correlations with WASI for ' group_name ' - Permutation Corrected'], 'FontSize', 16, 'FontWeight', 'bold');
end

function plot_dynamic_wasi_correlations_A(all_significant_results, group_name, data_type_names)
    fig = figure('Name', ['Alpha Power WASI Correlations (Dynamics) - ' group_name], 'WindowState', 'maximized');
    pnl = uipanel('Parent', fig, 'Title', data_type_names{1}, 'FontSize', 14, 'Position', [0.05 0.1 0.9 0.8]);
    if ~isempty(all_significant_results{1})
        t_nested = tiledlayout(pnl, 'flow', 'TileSpacing', 'compact', 'Padding', 'compact');
        for p = 1:length(all_significant_results{1})
            res = all_significant_results{1}{p}; nexttile(t_nested);
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
    ax_main_title = axes('Parent', fig, 'Position', [0 0 1 1], 'Visible', 'off');
    title(ax_main_title, ['Alpha Power Dynamic Correlations with WASI for ' group_name ' - Permutation Corrected'], 'FontSize', 16, 'FontWeight', 'bold');
end
