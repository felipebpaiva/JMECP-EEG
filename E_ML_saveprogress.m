%% SCRIPT 4: Machine Learning Prediction with Deep Learning and Permutation Testing
% This script implements a robust machine learning pipeline to predict WASI scores.
%
% CHANGES FOR STABILITY:
% - Implements "Chunked" Permutations (saves progress every N iterations).
% - Allows resuming/extending analysis without re-running everything.
% - Monitors p-value convergence.

%clear all;
%clc;
rng(123); 

%% --- 1. Load All Feature Data and Start Parallel Pool ---
disp('Loading all calculated features...');
% NOTE: Update paths if needed
%load('K:\JMECP_EEG_Analysis\results\RELAXv2\Topology_Features.mat');
%load('K:\JMECP_EEG_Analysis\results\RELAXv2\AlphaPower_Features.mat');
disp('All features loaded.');

% --- Start Parallel Pool ---
if isempty(gcp('nocreate'))
    parpool;
end
disp('Parallel pool started.');

% Check for GPU availability
useGPU = gpuDeviceCount("available") > 0;
if useGPU, disp('NVIDIA GPU detected. Training will be performed on the GPU.');
else, disp('No GPU detected. Training will be performed on the CPU.'); end


%% --- 2. Main Analysis Loop (Controls vs. Patients) ---
groups_to_analyze = {struct('Name', 'Controls', 'LabelIndex', 0), struct('Name', 'Patients', 'LabelIndex', 1)};

% PERMUTATION SETTINGS
n_target_perms = 10000; % Total desired permutations
chunk_size = 1000;      % Save and check p-value every 1000 runs

for g = 1:length(groups_to_analyze)
    current_group = groups_to_analyze{g};
    disp(['--- Starting analysis for: ' current_group.Name ' ---']);
    
    % --- 2A. Aggregate All Features for the Current Group ---
    group_indices = find(origLabels == current_group.LabelIndex);
    n_subjects_in_group = length(group_indices);
    Y_group = wasi(group_indices);
    
    all_feature_names = {};
    temp_feature_matrix = [];

    % Aggregate Static & Dynamic Topology Features
    feature_sets_topo = {features_0D, features_1D, features_0Dglobal, features_1Dglobal, dynamic_features_0D, dynamic_features_1D};
    set_names_topo = {'0D_trial_avg', '1D_trial_avg', '0D_global', '1D_global', '0D_dyn', '1D_dyn'};
    for s = 1:length(feature_sets_topo)
        f_names = fieldnames(feature_sets_topo{s});
        for f = 1:length(f_names)
            data = feature_sets_topo{s}.(f_names{f});
            if contains(set_names_topo{s}, 'trial'), subject_data = mean(data(:, group_indices), 1, 'omitnan');
            else, subject_data = data(group_indices); end
            temp_feature_matrix = [temp_feature_matrix; subject_data];
            all_feature_names{end+1} = [f_names{f} '_' set_names_topo{s}];
        end
    end
    
    % Aggregate Static & Dynamic Alpha Power Features
    feature_sets_alpha = {features_A, features_Aglobal, dynamic_features_AP};
    set_names_alpha = {'A_trial_avg', 'A_global', 'AP_dyn'};
    for s = 1:length(feature_sets_alpha)
        f_names = fieldnames(feature_sets_alpha{s});
        for f = 1:length(f_names)
            data = feature_sets_alpha{s}.(f_names{f});
            if contains(set_names_alpha{s}, 'trial'), subject_data = mean(data(:, group_indices), 1, 'omitnan');
            else, subject_data = data(group_indices); end
            temp_feature_matrix = [temp_feature_matrix; subject_data];
            all_feature_names{end+1} = [f_names{f} '_' set_names_alpha{s}];
        end
    end

    X_full = temp_feature_matrix';
    for i = 1:size(X_full, 2)
        col_mean = mean(X_full(:,i), 'omitnan'); if isnan(col_mean), col_mean = 0; end
        X_full(isnan(X_full(:,i)), i) = col_mean;
    end
    disp('Full feature aggregation complete.');

    % --- 2B. Run the "Real" Model (LOOCV once on unshuffled data) ---
    disp('Running the real model...');
    predicted_Y_real = nan(n_subjects_in_group, 1);
    feature_selection_counts = zeros(size(X_full, 2), 1);
    
    for i = 1:n_subjects_in_group
        train_idx = true(n_subjects_in_group, 1); train_idx(i) = false;
        X_train = X_full(train_idx, :); Y_train = Y_group(train_idx);
        X_test = X_full(i, :);

        [B, FitInfo] = lasso(X_train, Y_train, 'Alpha', 0.5, 'NumLambda', 25, 'CV', 5);
        idx = FitInfo.IndexMinMSE;
        selected_features_mask = (B(:, idx) ~= 0);
        
        if sum(selected_features_mask) == 0, predicted_Y_real(i) = mean(Y_train); continue; end
        
        feature_selection_counts(selected_features_mask) = feature_selection_counts(selected_features_mask) + 1;
        
        net = feedforwardnet(10);
        net.layers{1}.initFcn = 'initnw'; 
        net.trainFcn = 'trainscg'; 
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = false;
        
        if useGPU, net = train(net, X_train(:, selected_features_mask)', Y_train', 'useParallel', 'no', 'useGPU', 'yes');
        else, net = train(net, X_train(:, selected_features_mask)', Y_train', 'useParallel', 'no'); end
        predicted_Y_real(i) = net(X_test(:, selected_features_mask)');
    end
    
    real_r = corr(Y_group, predicted_Y_real, 'rows', 'complete');
    fprintf('Real Model Correlation: r = %.4f\n', real_r);

    % --- 2C. Run Permutations in Chunks (Resume/Extend Capability) ---
    perm_file = sprintf('Permutation_Results_%s.mat', current_group.Name);
    
    % Load existing progress if available
    if isfile(perm_file)
        fprintf('Found existing results file: %s. Resuming...\n', perm_file);
        load(perm_file, 'null_correlations');
    else
        null_correlations = [];
    end
    
    % Chunking Loop
    while length(null_correlations) < n_target_perms
        perms_done = length(null_correlations);
        perms_remaining = n_target_perms - perms_done;
        
        % Decide chunk size (don't overshoot target)
        current_chunk_size = min(chunk_size, perms_remaining);
        
        fprintf('Starting chunk of %d permutations (Total done: %d / %d)...\n', ...
            current_chunk_size, perms_done, n_target_perms);
        
        chunk_results = nan(current_chunk_size, 1);
        
        % Parallel Loop for the Chunk
        parfor p = 1:current_chunk_size
            % Use a local stream or re-seed based on iter to ensure randomness 
            % across resumed sessions if needed, but randperm usually suffices.
            Y_perm = Y_group(randperm(n_subjects_in_group));
            predicted_Y_perm = nan(n_subjects_in_group, 1);
            
            for i = 1:n_subjects_in_group
                train_idx = true(n_subjects_in_group, 1); train_idx(i) = false;
                X_train = X_full(train_idx, :); Y_train_perm = Y_perm(train_idx);
                X_test = X_full(i, :);
                
                [B, FitInfo] = lasso(X_train, Y_train_perm, 'Alpha', 0.5, 'NumLambda', 25, 'CV', 5);
                idx = FitInfo.IndexMinMSE;
                mask = (B(:, idx) ~= 0);
                
                if sum(mask) == 0
                    predicted_Y_perm(i) = mean(Y_train_perm); 
                    continue; 
                end
                
                net = feedforwardnet(10); 
                net.layers{1}.initFcn = 'initnw';
                net.trainFcn = 'trainscg';
                net.trainParam.showWindow = false;
                net.trainParam.showCommandLine = false;

                if useGPU
                    net = train(net, X_train(:, mask)', Y_train_perm', 'useParallel', 'no', 'useGPU', 'yes');
                else
                    net = train(net, X_train(:, mask)', Y_train_perm', 'useParallel', 'no'); 
                end
                predicted_Y_perm(i) = net(X_test(:, mask)');
            end
            chunk_results(p) = corr(Y_perm, predicted_Y_perm, 'rows', 'complete');
        end
        
        % Append and Save
        null_correlations = [null_correlations; chunk_results];
        save(perm_file, 'null_correlations', 'real_r');
        
        % Convergence Check
        curr_p_val = (sum(null_correlations >= real_r) + 1) / (length(null_correlations) + 1);
        fprintf('  Chunk complete. Saved. Current p-value (N=%d): %.4f\n', ...
            length(null_correlations), curr_p_val);
    end
    
    % --- 2D. Final Results ---
    p_value = (sum(null_correlations >= real_r) + 1) / (length(null_correlations) + 1);

    fprintf('\n--- Final Model Performance for %s ---\n', current_group.Name);
    fprintf('Prediction Correlation: r = %.3f\n', real_r);
    fprintf('Permutation p-value: p = %.4f (N=%d)\n', p_value, length(null_correlations));

    % --- 2E. Generate Plots ---
    figure('Name', ['Model Performance - ' current_group.Name]);
    scatter(Y_group, predicted_Y_real, 100, 'filled', 'MarkerFaceAlpha', 0.7);
    hold on;
    plot([min(Y_group), max(Y_group)], [min(Y_group), max(Y_group)], 'r--', 'LineWidth', 2);
    xlabel('Actual WASI Score'); ylabel('Predicted WASI Score');
    title(sprintf('WASI Score Prediction for %s (r = %.3f, p = %.4f)', current_group.Name, real_r, p_value));
    grid on; legend('Predicted Score', 'Perfect Prediction', 'Location', 'northwest');

    figure('Name', ['Feature Importance - ' current_group.Name]);
    [sorted_counts, sorted_idx] = sort(feature_selection_counts, 'descend');
    num_features_to_plot = min(20, sum(sorted_counts > 0));
    barh(sorted_counts(1:num_features_to_plot));
    set(gca, 'YTick', 1:num_features_to_plot, 'YTickLabel', strrep(all_feature_names(sorted_idx(1:num_features_to_plot)), '_', ' '));
    set(gca, 'YDir','reverse');
    xlabel('Number of Times Selected in LOOCV'); title('Predictive Feature Importance');
    grid on;
    
    % Plot Null Distribution
    figure('Name', ['Null Distribution - ' current_group.Name]);
    histogram(null_correlations, 50, 'Normalization', 'probability');
    xline(real_r, 'r--', 'LineWidth', 2);
    title(sprintf('Null Distribution (N=%d)\nReal r=%.3f, p=%.4f', length(null_correlations), real_r, p_value));
    xlabel('Correlation (r)'); ylabel('Probability');
end