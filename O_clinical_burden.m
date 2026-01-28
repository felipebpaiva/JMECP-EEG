%% SCRIPT 22: Clinical Correlation (Epileptiform Prevalence vs. Stiffness)
% This script tests the "Double-Edged Sword" Hypothesis:
% Does the "Loosening" (Low Stiffness) that benefits cognition also 
% correlate with increased Epileptiform Activity?
%
% DATA SOURCE:
% Excel: JME_07222024_psych_cog_clusters_g_psych_clusters_hdEEGvars.xlsx
% - Column B: Subject Code
% - Column WM: Epileptiform Prevalence Score (0-4)
%
% INPUTS:
% - results_patients (from Script 19/20)
% - subcode (Workspace variable with subject IDs)
% - origLabels (Workspace variable with group labels)

clc; %close all;

% Ensure results exist
if ~exist('results_patients', 'var')
    error('Please run Script 19 first to generate results_patients.');
end

%% --- 1. Load Clinical Data ---
disp('Loading Epileptiform Prevalence Data...');

spreadsheet_path = 'K:\JMECP_EEG_Analysis\JME_07222024_psych_cog_clusters_g_psych_clusters_hdEEGvars.xlsx';

% Read Subject Codes (Column B)
try
    code_table = readtable(spreadsheet_path, 'Range', 'B2:B141', 'ReadVariableNames', false);
    clinical_codes = table2array(code_table);
    
    % Read Prevalence (Column WM)
    prev_table = readtable(spreadsheet_path, 'Range', 'WM2:WM141', 'ReadVariableNames', false);
    clinical_prev = table2array(prev_table);
    
catch ME
    error('Failed to read Excel file. Check path and close file if open. Error: %s', ME.message);
end

%% --- 2. Match Data to Patients ---
if ~exist('subcode', 'var') || ~exist('origLabels', 'var')
    error('Variable "subcode" or "origLabels" missing from workspace.');
end

patient_indices = find(origLabels == 1);
patient_ids = subcode(patient_indices);

% Extract Metrics
pat_stiff = results_patients(:, 2);
pat_temp  = results_patients(:, 3);
pat_wasi  = results_patients(:, 1);

matched_prev = nan(length(patient_ids), 1);

for i = 1:length(patient_ids)
    pid = patient_ids(i);
    idx = find(clinical_codes == pid);
    if ~isempty(idx)
        matched_prev(i) = clinical_prev(idx(1));
    end
end

% Initial Filter (Valid Data)
valid_mask = ~isnan(matched_prev);
final_stiff = pat_stiff(valid_mask);
final_temp  = pat_temp(valid_mask);
final_prev  = matched_prev(valid_mask);
final_wasi  = pat_wasi(valid_mask);

%% --- 3. Outlier Rejection (MAD Method) ---
% Remove extreme outliers (> 3 Median Absolute Deviations)
% This ensures result is not driven by the single -40 stiffness point.

is_outlier = isoutlier(final_stiff, 'median') | isoutlier(final_temp, 'median');
final_stiff_clean = final_stiff(~is_outlier);
final_temp_clean  = final_temp(~is_outlier);
final_prev_clean  = final_prev(~is_outlier);
final_wasi_clean  = final_wasi(~is_outlier);

fprintf('Matched %d patients.\n', length(final_stiff));
fprintf('Removed %d outliers based on Stiffness/Temp distribution.\n', sum(is_outlier));
fprintf('Analyzing N=%d clean subjects.\n', length(final_stiff_clean));

%% --- 4. Plotting & Statistics ---
fig = figure('Name', 'Epileptiform Prevalence Analysis (Cleaned)', 'WindowState', 'maximized');
t = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Plot 1: Prevalence vs. Stiffness
% Hypothesis: Higher Prevalence (More Epilepsy) -> Lower Stiffness (Looser)
nexttile;
plot_clinical_corr(final_prev_clean, final_stiff_clean, 'Stiffness (Curvature)', 'r', ...
    'Hypothesis 1: Does Epilepsy "Loosen" the Brain?');

% Plot 2: Prevalence vs. Temperature
% Hypothesis: Higher Prevalence -> Higher Noise?
nexttile;
plot_clinical_corr(final_prev_clean, final_temp_clean, 'Temperature (Diffusion)', 'm', ...
    'Hypothesis 2: Is Epilepsy Noisier?');

% Plot 3: Boxplot of Stiffness by Prevalence Score
nexttile;
boxplot(final_stiff_clean, final_prev_clean, 'Colors', 'k', 'Symbol', '+');
xlabel('Epileptiform Prevalence Score (0-4)');
ylabel('Stiffness (b)');
title('Rigidity vs. Clinical Severity');
grid on;

% --- 5. INTERACTION ANALYSIS (The "Overcare" Hypothesis) ---
idx_low_prev = final_prev_clean <= 1;
idx_high_prev = final_prev_clean > 1;

% Plot 4: Subgroup Slopes
nexttile; hold on;
plot_subgroup_scatter(final_stiff_clean(idx_low_prev), final_wasi_clean(idx_low_prev), 'Low Prev (0-1)', 'b');
plot_subgroup_scatter(final_stiff_clean(idx_high_prev), final_wasi_clean(idx_high_prev), 'High Prev (2-4)', 'r');
title('Does Clinical Context Moderate the Stiffness Cost?');
xlabel('Stiffness (Z)'); ylabel('WASI'); 
legend('Location', 'best'); grid on;

% --- 6. Formal Interaction Model ---
nexttile([1,2]); 
axis off;

% Normalize
z_stiff = (final_stiff_clean - mean(final_stiff_clean))/std(final_stiff_clean);
z_prev  = (final_prev_clean - mean(final_prev_clean))/std(final_prev_clean);
z_wasi  = (final_wasi_clean - mean(final_wasi_clean))/std(final_wasi_clean);

tbl = table(z_stiff, z_prev, z_wasi, 'VariableNames', {'Stiffness', 'Prevalence', 'WASI'});
mdl = fitlm(tbl, 'WASI ~ Stiffness * Prevalence');

coeffs = mdl.Coefficients;
beta_stiff = coeffs.Estimate(2);
p_stiff = coeffs.pValue(2);
beta_inter = coeffs.Estimate(4);
p_inter = coeffs.pValue(4);

text(0, 0.8, 'Formal Interaction Test (GLM - Outliers Removed)', 'FontSize', 12, 'FontWeight', 'bold');
text(0, 0.6, sprintf('Model: WASI ~ Stiffness + Prevalence + (Stiffness x Prevalence)'), 'FontSize', 11);
text(0, 0.4, sprintf('Stiffness Main Effect: \\beta = %.3f (p=%.3f)', beta_stiff, p_stiff), 'FontSize', 11);
text(0, 0.2, sprintf('Interaction Term:      \\beta = %.3f (p=%.3f)', beta_inter, p_inter), 'FontSize', 11, 'Color', 'r');

if p_inter < 0.1
    if beta_inter > 0
        interpretation = 'RESULT: Positive Interaction. High Prev PROTECTS against Stiffness cost.';
    else
        interpretation = 'RESULT: Negative Interaction. High Prev WORSENS Stiffness cost.';
    end
else
    interpretation = 'RESULT: No significant interaction.';
end
text(0, 0.0, interpretation, 'FontSize', 12, 'FontWeight', 'bold');


%% --- Helper Functions ---
function plot_clinical_corr(x, y, y_label, color_c, main_title)
    x_jit = x + (rand(size(x))-0.5)*0.2;
    scatter(x_jit, y, 60, color_c, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    try
        [b, stats] = robustfit(x, y);
        x_grid = linspace(min(x), max(x), 100)';
        plot(x_grid, b(1) + b(2)*x_grid, 'Color', color_c, 'LineWidth', 2);
        x_z = (x - mean(x)) / std(x);
        y_z = (y - mean(y)) / std(y);
        [b_std, stats_std] = robustfit(x_z, y_z);
        beta = b_std(2); p_val = stats_std.p(2);
    catch
        p_val = NaN; beta = NaN;
    end
    [rho, p_rho] = corr(x, y, 'Type', 'Spearman', 'Rows', 'complete');
    title(sprintf('%s\nSpearman \\rho=%.2f (p=%.3f) | Robust \\beta=%.2f (p=%.3f)', ...
        main_title, rho, p_rho, beta, p_val), 'FontSize', 10);
    xlabel('Epileptiform Prevalence (0-4)');
    ylabel(y_label);
    grid on;
end

function plot_subgroup_scatter(x, y, label, color_c)
    scatter(x, y, 50, color_c, 'filled', 'MarkerFaceAlpha', 0.6, 'DisplayName', label);
    if length(x) > 3
        [b, stats] = robustfit(x, y);
        x_grid = linspace(min(x), max(x), 100)';
        plot(x_grid, b(1) + b(2)*x_grid, 'Color', color_c, 'LineWidth', 2, ...
            'DisplayName', sprintf('Fit %s (p=%.3f)', label, stats.p(2)));
    end
end