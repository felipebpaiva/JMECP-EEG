%% SCRIPT 27: Generate Final ML Publication Figure
% This script reconstructs the ML model results for visualization WITHOUT
% re-running the expensive permutation tests.
%
% PROCESS:
% 1. Re-runs the "Fast" LOOCV (seconds/minutes) to get predictions & features.
% 2. Loads the "Slow" Permutation results from saved .mat files.
% 3. Generates a composite 2x3 figure (Controls vs Patients).

clc; 
% close all; % Optional: Keep existing figures open
rng(123); % Ensure MLP initialization matches original run

%% --- 1. Check/Load Features ---
% We need the feature variables (features_0D, features_A, etc.)
if ~exist('features_0D', 'var') || ~exist('wasi', 'var')
    disp('Feature variables not found in workspace.');
    disp('Please load "Topology_Features.mat" and "AlphaPower_Features.mat".');
    % Uncomment to load automatically if paths are known:
    % load('K:\JMECP_EEG_Analysis\results\RELAXv2\Topology_Features.mat');
    % load('K:\JMECP_EEG_Analysis\results\RELAXv2\AlphaPower_Features.mat');
    error('Stopping: Load feature data first.');
end

%% --- 2. Reconstruction Loop ---
groups = {struct('Name', 'Controls', 'LabelIndex', 0), struct('Name', 'Patients', 'LabelIndex', 1)};
results_store = struct();

disp('Reconstructing model outputs (Fast LOOCV)...');

for g = 1:length(groups)
    curr_grp = groups{g};
    fprintf('  Processing %s...\n', curr_grp.Name);
    
    % --- A. Aggregate Features (Same logic as Script 4) ---
    g_idx = find(origLabels == curr_grp.LabelIndex);
    n_sub = length(g_idx);
    Y_act = wasi(g_idx);
    
    % Aggregate Topology
    sets_topo = {features_0D, features_1D, features_0Dglobal, features_1Dglobal, dynamic_features_0D, dynamic_features_1D};
    names_topo = {'0D_trial', '1D_trial', '0D_glob', '1D_glob', '0D_dyn', '1D_dyn'};
    
    X_temp = [];
    feat_names = {};
    
    for s = 1:length(sets_topo)
        f_fields = fieldnames(sets_topo{s});
        for f = 1:length(f_fields)
            dat = sets_topo{s}.(f_fields{f});
            if contains(names_topo{s}, 'trial'), sub_dat = mean(dat(:, g_idx), 1, 'omitnan');
            else, sub_dat = dat(g_idx); end
            X_temp = [X_temp; sub_dat];
            feat_names{end+1} = [f_fields{f} '_' names_topo{s}];
        end
    end
    
    % Aggregate Alpha
    sets_alpha = {features_A, features_Aglobal, dynamic_features_AP};
    names_alpha = {'A_trial', 'A_glob', 'AP_dyn'};
    for s = 1:length(sets_alpha)
        f_fields = fieldnames(sets_alpha{s});
        for f = 1:length(f_fields)
            dat = sets_alpha{s}.(f_fields{f});
            if contains(names_alpha{s}, 'trial'), sub_dat = mean(dat(:, g_idx), 1, 'omitnan');
            else, sub_dat = dat(g_idx); end
            X_temp = [X_temp; sub_dat];
            feat_names{end+1} = [f_fields{f} '_' names_alpha{s}];
        end
    end
    
    X_full = X_temp';
    % Impute NaNs with col mean
    for i = 1:size(X_full, 2)
        m = mean(X_full(:,i), 'omitnan'); if isnan(m), m=0; end
        X_full(isnan(X_full(:,i)), i) = m;
    end
    
    % --- B. Run LOOCV (The Fast Part) ---
    Y_pred = nan(n_sub, 1);
    feat_counts = zeros(size(X_full, 2), 1);
    
    % Check GPU
    useGPU = gpuDeviceCount > 0;
    
    for i = 1:n_sub
        train_mask = true(n_sub, 1); train_mask(i) = false;
        X_tr = X_full(train_mask, :); Y_tr = Y_act(train_mask);
        X_te = X_full(i, :);
        
        [B, Info] = lasso(X_tr, Y_tr, 'Alpha', 0.5, 'NumLambda', 25, 'CV', 5);
        idx_best = Info.IndexMinMSE;
        mask_sel = (B(:, idx_best) ~= 0);
        
        if sum(mask_sel) == 0, Y_pred(i) = mean(Y_tr); continue; end
        
        feat_counts(mask_sel) = feat_counts(mask_sel) + 1;
        
        net = feedforwardnet(10);
        net.layers{1}.initFcn = 'initnw';
        net.trainFcn = 'trainscg';
        net.trainParam.showWindow = false; net.trainParam.showCommandLine = false;
        
        if useGPU
            net = train(net, X_tr(:, mask_sel)', Y_tr', 'useParallel', 'no', 'useGPU', 'yes');
        else
            net = train(net, X_tr(:, mask_sel)', Y_tr', 'useParallel', 'no');
        end
        Y_pred(i) = net(X_te(:, mask_sel)');
    end
    
    real_r = corr(Y_act, Y_pred, 'rows', 'complete');
    
    % --- C. Load Saved Permutations (The Slow Part) ---
    perm_file = sprintf('Permutation_Results_%s.mat', curr_grp.Name);
    if isfile(perm_file)
        pd = load(perm_file);
        null_dist = pd.null_correlations;
        p_val = (sum(null_dist >= real_r) + 1) / (length(null_dist) + 1);
        fprintf('  Loaded %d permutations. P-value = %.4f\n', length(null_dist), p_val);
    else
        warning('Permutation file %s not found! Using dummy p-value.', perm_file);
        null_dist = [];
        p_val = NaN;
    end
    
    % Store
    results_store(g).Name = curr_grp.Name;
    results_store(g).Y_act = Y_act;
    results_store(g).Y_pred = Y_pred;
    results_store(g).FeatCounts = feat_counts;
    results_store(g).FeatNames = feat_names;
    results_store(g).NullDist = null_dist;
    results_store(g).Stats = [real_r, p_val];
end

%% --- 3. Generate Publication Figure ---
fig = figure('Name', 'ML Performance Composite', 'Color', 'w', 'Units', 'normalized', 'Position', [0.1 0.1 0.7 0.8]);
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

colors = {'b', 'r'}; % Blue=Control, Red=Patient

for g = 1:2
    res = results_store(g);
    c = colors{g};
    
    % --- Panel 1: Prediction Scatter ---
    nexttile;
    scatter(res.Y_act, res.Y_pred, 60, c, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    
    % Bounds
    all_vals = [res.Y_act; res.Y_pred];
    mi = min(all_vals); ma = max(all_vals);
    plot([mi ma], [mi ma], 'k--', 'LineWidth', 1.5); % Identity
    h = lsline; set(h, 'Color', c, 'LineWidth', 2); % Trend
    
    xlabel('Actual WASI'); ylabel('Predicted WASI');
    title(sprintf('%s Prediction\nr=%.3f, p=%.4f', res.Name, res.Stats(1), res.Stats(2)));
    grid on; axis square;
    
    % --- Panel 2: Feature Importance ---
    nexttile;
    [sorted_c, sorted_idx] = sort(res.FeatCounts, 'descend');
    n_plot = min(15, sum(sorted_c > 0));
    
    if n_plot > 0
        barh(sorted_c(1:n_plot), 'FaceColor', c, 'EdgeColor', 'none');
        yticks(1:n_plot);
        % Clean names for display
        lbls = res.FeatNames(sorted_idx(1:n_plot));
        lbls = strrep(lbls, '_', ' ');
        yticklabels(lbls);
        set(gca, 'YDir', 'reverse', 'FontSize', 8);
    else
        text(0.5, 0.5, 'No Features Selected', 'HorizontalAlignment', 'center');
        axis off;
    end
    title('Top Features (LOOCV Selection)'); xlabel('Selection Count');
    grid on;
    
    % --- Panel 3: Null Distribution ---
    nexttile;
    if ~isempty(res.NullDist)
        histogram(res.NullDist, 50, 'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'none', 'Normalization', 'probability');
        hold on;
        xline(res.Stats(1), 'Color', c, 'LineWidth', 2, 'LineStyle', '--');
        xlabel('Correlation (r)'); ylabel('Probability');
        title(sprintf('Null Distribution (N=%d)', length(res.NullDist)));
        legend('Null', 'Real', 'Location', 'northwest');
    else
        text(0.5, 0.5, 'Permutations Not Found', 'HorizontalAlignment', 'center');
    end
    grid on; axis square;
end