%% SCRIPT 7: Machine Learning Prediction using SDI Features
% This script adapts the ML pipeline (Script 4) to predict WASI scores
% using the spatial pattern of the Structural Decoupling Index (SDI) across
% electrodes as input features.
% 1. Loads the calculated SDI maps ('sdi_log2').
% 2. Uses SDI values at each electrode as features.
% 3. Employs LOOCV with Elastic Net for feature selection and MLP for prediction.
% 4. Validates the model performance using parallelized permutation testing.

clear all;
clc;

%% --- 1. Load SDI Feature Data and Start Parallel Pool ---
disp('Loading SDI feature data...');
% NOTE: Update file path if necessary
load('K:\JMECP_EEG_Analysis\results\RELAXv2\SDI_Results_TimeSeries_DynamicGraphs.mat', 'sdi_log2', 'origLabels', 'wasi', 'Nroi', 'nsubj');
disp('SDI features loaded.');

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
n_perms = 1000;

for g = 1:length(groups_to_analyze)
    current_group = groups_to_analyze{g};
    disp(['--- Starting analysis for: ' current_group.Name ' ---']);
    
    % --- 2A. Prepare Feature Matrix for the Current Group ---
    group_indices = find(origLabels == current_group.LabelIndex);
    n_subjects_in_group = length(group_indices);
    Y_group = wasi(group_indices);
    
    % Feature matrix is SDI values: subjects x electrodes
    X_full = sdi_log2(:, group_indices)'; 
    
    % Create feature names (Electrode 1, Electrode 2, ...)
    all_feature_names = cell(1, Nroi);
    for i = 1:Nroi
        all_feature_names{i} = ['Electrode ' num2str(i)];
    end
    
    % Handle potential NaNs in SDI data (column-wise mean imputation)
    for i = 1:size(X_full, 2)
        col_data = X_full(:, i);
        nan_mask = isnan(col_data);
        if any(nan_mask)
            col_mean = mean(col_data, 'omitnan');
            if isnan(col_mean), col_mean = 0; end % Handle case where whole column might be NaN
            X_full(nan_mask, i) = col_mean;
        end
    end
    disp('SDI feature matrix prepared.');

    % --- 2B. Run the "Real" Model (LOOCV once on unshuffled data) ---
    disp('Running the real model...');
    predicted_Y_real = nan(n_subjects_in_group, 1);
    feature_selection_counts = zeros(size(X_full, 2), 1);
    
    for i = 1:n_subjects_in_group
        train_idx = true(n_subjects_in_group, 1); train_idx(i) = false;
        X_train = X_full(train_idx, :); Y_train = Y_group(train_idx);
        X_test = X_full(i, :);

        % Feature Selection with Elastic Net
        [B, FitInfo] = lasso(X_train, Y_train, 'Alpha', 0.5, 'NumLambda', 25, 'CV', 5);
        idx = FitInfo.IndexMinMSE;
        selected_features_mask = (B(:, idx) ~= 0);
        
        if sum(selected_features_mask) == 0, predicted_Y_real(i) = mean(Y_train); continue; end
        
        feature_selection_counts(selected_features_mask) = feature_selection_counts(selected_features_mask) + 1;
        
        % Train MLP on selected features
        net = feedforwardnet(10); 
        net.trainFcn = 'trainscg'; 
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = false;
        
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

            if useGPU, net = train(net, X_train(:, selected_features_mask)', Y_train_perm', 'useParallel', 'no', 'useGPU', 'yes');
            else, net = train(net, X_train(:, selected_features_mask)', Y_train_perm', 'useParallel', 'no'); end
            predicted_Y_perm(i) = net(X_test(:, selected_features_mask)');
        end
        % Handle potential NaNs in predictions if imputation failed for a fold
        valid_perm_preds = ~isnan(predicted_Y_perm);
        if sum(valid_perm_preds) > 2
             null_correlations(p) = corr(Y_perm(valid_perm_preds), predicted_Y_perm(valid_perm_preds), 'rows', 'complete');
        else
             null_correlations(p) = 0; % Assign 0 if correlation cannot be computed
        end
    end
    
    % --- 2D. Calculate Final p-value and Display Results ---
    p_value = (sum(null_correlations >= real_r) + 1) / (n_perms + 1);

    fprintf('\n--- Model Performance for %s (using SDI features) ---\n', current_group.Name);
    fprintf('Prediction Correlation: r = %.3f\n', real_r);
    fprintf('Permutation p-value: p = %.4f\n', p_value);

    % --- 2E. Generate Plots for the Real Model ---
    figure('Name', ['SDI Model Performance - ' current_group.Name]);
    scatter(Y_group, predicted_Y_real, 100, 'filled', 'MarkerFaceAlpha', 0.7);
    hold on;
    plot([min(Y_group), max(Y_group)], [min(Y_group), max(Y_group)], 'r--', 'LineWidth', 2);
    xlabel('Actual WASI Score'); ylabel('Predicted WASI Score');
    title(sprintf('WASI Score Prediction from SDI for %s (r = %.3f, p = %.4f)', current_group.Name, real_r, p_value));
    grid on; legend('Predicted Score', 'Perfect Prediction', 'Location', 'northwest');

    figure('Name', ['SDI Feature Importance - ' current_group.Name]);
    [sorted_counts, sorted_idx] = sort(feature_selection_counts, 'descend');
    num_features_to_plot = min(20, sum(sorted_counts > 0));
    % Ensure indices are valid before plotting
    valid_plot_indices = sorted_idx(1:num_features_to_plot);
    valid_counts = sorted_counts(1:num_features_to_plot);
    valid_names = all_feature_names(valid_plot_indices);

    barh(valid_counts);
    set(gca, 'YTick', 1:num_features_to_plot, 'YTickLabel', strrep(valid_names, '_', ' '));
    set(gca, 'YDir','reverse');
    xlabel('Number of Times Selected in LOOCV'); title('Predictive Feature Importance (SDI Electrodes)');
    grid on;
end
