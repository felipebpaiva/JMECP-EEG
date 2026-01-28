%% SCRIPT 26: Permutation Convergence Analysis
% This script analyzes the stability of the p-value over the course of 
% the permutation test to demonstrate convergence.
%
% METHOD:
% 1. Loads 'Permutation_Results_Controls.mat'
% 2. Calculates the cumulative p-value for N = 1 to Total.
% 3. Calculates the 95% Confidence Interval (Binomial Proportion) at each N.
% 4. Plots the trajectory to show stability.

clc; %close all;

%% --- 1. Load Data ---
% Update filename if specific path needed
file_name = 'Permutation_Results_Controls.mat'; 

if ~isfile(file_name)
    error('Results file %s not found. Run Script 4 first.', file_name);
end

data = load(file_name);
null_r = data.null_correlations;
real_r = data.real_r;

N_total = length(null_r);
fprintf('Loaded %d permutations. Real r = %.4f\n', N_total, real_r);

%% --- 2. Calculate Cumulative Statistics ---
n_steps = 1:N_total;
p_values = zeros(N_total, 1);
ci_upper = zeros(N_total, 1);
ci_lower = zeros(N_total, 1);

% Vectorized Calculation for speed
% Count how many nulls >= real (Cumulative Sum)
hits = cumsum(null_r >= real_r);

% Calculate P-value at each step N
% Formula: (Hits + 1) / (N + 1)  (Standard conservative estimator)
p_values = (hits + 1) ./ (n_steps' + 1);

% Calculate Standard Error and 95% CI (Wald Interval)
% SE = sqrt( p*(1-p) / N )
se = sqrt(p_values .* (1 - p_values) ./ n_steps');
margin_error = 1.96 * se;

ci_upper = p_values + margin_error;
ci_lower = p_values - margin_error;

%% --- 3. Plotting Convergence ---
fig = figure('Name', 'Permutation Convergence', 'Color', 'w', 'Position', [100, 100, 1000, 500]);
t = tiledlayout(1, 2, 'Padding', 'compact');

% Plot A: P-value Evolution
nexttile;
hold on;
% Plot CI as shaded region
fill([n_steps, fliplr(n_steps)], [ci_upper', fliplr(ci_lower')], ...
    [0.8 0.8 0.8], 'EdgeColor', 'none', 'DisplayName', '95% Confidence Interval');

% Plot P-value trace
plot(n_steps, p_values, 'k-', 'LineWidth', 1.5, 'DisplayName', 'Cumulative p-value');

% Add markers every 1000
marker_idx = 1000:1000:N_total;
if ~isempty(marker_idx)
    plot(n_steps(marker_idx), p_values(marker_idx), 'ro', 'MarkerFaceColor', 'r', 'DisplayName', 'Checkpoints (1k)');
end

% Reference line at 0.05
yline(0.05, 'b--', 'Significance \alpha=0.05', 'DisplayName', 'Significance \alpha=0.05');

xlabel('Number of Permutations');
ylabel('Estimated p-value');
title(sprintf('P-value Convergence (Final p=%.4f)', p_values(end)));
legend('Location', 'northeast');
grid on;
ylim([min(ci_lower(1000:end)) max(ci_upper(1000:end))*1.2]); % Zoom y to relevant range (skip unstable start)
xlim([0 N_total]);

% Plot B: Margin of Error (Precision)
nexttile;
plot(n_steps, margin_error, 'b-', 'LineWidth', 1.5);
yline(0.005, 'r--', 'High Precision Threshold (\pm0.005)');
xlabel('Number of Permutations');
ylabel('Margin of Error (95% CI)');
title('Statistical Precision');
grid on; xlim([1000 N_total]);

%% --- 4. Formal Convergence Check ---
% Check stability over last 10% of runs
window = floor(0.1 * N_total);
recent_p = p_values(end-window:end);
p_range = range(recent_p);
final_margin = margin_error(end);

fprintf('\n--- CONVERGENCE REPORT ---\n');
fprintf('Final P-value: %.5f\n', p_values(end));
fprintf('95%% CI:        [%.5f, %.5f]\n', ci_lower(end), ci_upper(end));
fprintf('Margin of Error: +/- %.5f\n', final_margin);
fprintf('Fluctuation (last %d runs): %.5f\n', window, p_range);

if final_margin < 0.01
    fprintf('STATUS: CONVERGED. Precision is high (< 1%% error).\n');
    fprintf('Conclusion: Stopping at %d permutations is statistically justified.\n', N_total);
else
    fprintf('STATUS: UNSTABLE. Margin of error > 1%%. Consider running more permutations.\n');
end