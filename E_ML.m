%% SCRIPT 4: Machine Learning Prediction with Deep Learning and Permutation Testing
% This script implements a robust machine learning pipeline to predict WASI scores.
% 1. Aggregates all previously calculated features.
% 2. Uses a Leave-One-Out Cross-Validation (LOOCV) framework.
% 3. Inside each fold, an Elastic Net (a more robust form of LASSO) is used
%    for automatic feature selection on the training data.
% 4. A simple feedforward neural network (MLP) is then trained on these selected
%    features to predict the WASI score of the left-out subject.
% 5. The overall model performance is validated using a parallelized permutation
%    test to derive a robust p-value.

%clear all;
clc;

%% --- 1. Load All Feature Data and Start Parallel Pool ---
disp('Loading all calculated features...');
% NOTE: Update these file paths if you saved the feature data elsewhere
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
n_perms = 250;

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
        net.trainFcn = 'trainscg'; 
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = false;
        
        % **FIX**: The 'useParallel' option is critical to prevent crashes inside parfor
        if useGPU, net = train(net, X_train(:, selected_features_mask)', Y_train', 'useParallel', 'no', 'useGPU', 'yes');
        else, net = train(net, X_train(:, selected_features_mask)', Y_train', 'useParallel', 'no'); end
        predicted_Y_real(i) = net(X_test(:, selected_features_mask)');
    end
    
    real_r = corr(Y_group, predicted_Y_real, 'rows', 'complete');

    % --- 2C. Run Permutations in Parallel ---
    disp('Running permutations in parallel...');
    null_correlations = nan(n_perms, 1);
    
    parfor p = 1:n_perms
        Y_perm = Y_group(randperm(n_subjects_in_group));
        predicted_Y_perm = nan(n_subjects_in_group, 1);
        
        for i = 1:n_subjects_in_group
            train_idx = true(n_subjects_in_group, 1); train_idx(i) = false;
            X_train = X_full(train_idx, :); Y_train_perm = Y_perm(train_idx);
            X_test = X_full(i, :);
            
            [B, FitInfo] = lasso(X_train, Y_train_perm, 'Alpha', 0.5, 'NumLambda', 25, 'CV', 5);
            idx = FitInfo.IndexMinMSE;
            selected_features_mask = (B(:, idx) ~= 0);
            
            if sum(selected_features_mask) == 0, predicted_Y_perm(i) = mean(Y_train_perm); continue; end
            
            net = feedforwardnet(10); 
            net.trainFcn = 'trainscg';
            net.trainParam.showWindow = false;
            net.trainParam.showCommandLine = false;

            % **FIX**: The 'useParallel' option is critical to prevent crashes
            if useGPU, net = train(net, X_train(:, selected_features_mask)', Y_train_perm', 'useParallel', 'no', 'useGPU', 'yes');
            else, net = train(net, X_train(:, selected_features_mask)', Y_train_perm', 'useParallel', 'no'); end
            predicted_Y_perm(i) = net(X_test(:, selected_features_mask)');
        end
        null_correlations(p) = corr(Y_perm, predicted_Y_perm, 'rows', 'complete');
    end
    
    % --- 2D. Calculate Final p-value and Display Results ---
    p_value = (sum(null_correlations >= real_r) + 1) / (n_perms + 1);

    fprintf('\n--- Model Performance for %s ---\n', current_group.Name);
    fprintf('Prediction Correlation: r = %.3f\n', real_r);
    fprintf('Permutation p-value: p = %.4f\n', p_value);

    % --- 2E. Generate Plots for the Real Model ---
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
end

