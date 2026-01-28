%% SCRIPT 19: In-Silico Mechanism Comparison (Langevin)
% This script applies Langevin Inverse Modeling to the Control model and 
% 3 distinct JME hypotheses (Untreated Baselines).
%
% GOAL: Identify which microscopic abnormality produces the "Cognitive Deficit"
% phenotype observed in low-WASI patients:
%   TARGET PHENOTYPE: High Stiffness (Rigidity) + High Temperature (Noise)
%
% HYPOTHESES:
% 0. Control (Baseline, 30s)
% 1. Poor Arborization (Reduced Distal Drive)
% 2. GABA Dysfunction (Reduced Inhibition)
% 3. Microdysgenesis (Increased Local Recurrence)

clc; %close all;

%% --- 1. Setup Parameters ---
dt_sim = 1/100; 
num_bins = 20; % 20 bins is safe for 30s data (3000 points)
x_axis_common = linspace(-3, 3, num_bins); 

% Files to analyze (Matched to Python output names)
files = {'InSilico_Control.mat', ...
         'InSilico_Arbor_Untreated.mat', ...
         'InSilico_GABA_Untreated.mat', ...
         'InSilico_Micro_Untreated.mat'};

% Display Labels
labels = {'Control', ...
          'Poor Arborization', ...
          'GABA Dysfunction', ...
          'Microdysgenesis'};

% Colors (Control=Black, Others=Colored)
colors = {[0 0 0], ...          % Black
          [0.2 0.6 0.8], ...    % Teal
          [0.8 0.2 0.8], ...    % Purple
          [0.8 0.4 0.2]};       % Orange

results = []; % [Stiffness(b), Temperature(D)]
potentials = []; 

%% --- 2. Simulation Loop ---
disp('Analyzing In-Silico Mechanisms...');

for i = 1:length(files)
    filename = files{i};
    if ~isfile(filename)
        warning('File %s not found. Run Python simulation scripts first.', filename);
        results = [results; NaN, NaN];
        potentials = [potentials; nan(1, num_bins)];
        continue;
    end
    
    fprintf('Processing %s...\n', labels{i});
    data = load(filename);
    
    A = double(data.A_dyn);
    X = double(data.elects);
    [N, ~] = size(X);
    
    % --- Global Smoothness ---
    A(isnan(A)) = 0;
    D_mat = diag(sum(A, 2));
    d_inv = 1./sqrt(diag(D_mat) + eps);
    L = eye(N) - diag(d_inv) * A * diag(d_inv);
    
    P = X.^2; % Alpha Power
    smooth_ts = sum(P .* (L * P), 1);
    
    % --- Langevin Reconstruction ---
    x_state = (smooth_ts - mean(smooth_ts)) / std(smooth_ts);
    dx = diff(x_state);
    
    % Adaptive Binning
    limits = prctile(x_state, [2.5, 97.5]);
    if abs(limits(2)-limits(1)) < 0.1, limits = [-3, 3]; end
    edges = linspace(limits(1), limits(2), num_bins+1);
    centers = (edges(1:end-1) + edges(2:end)) / 2;
    [~, bin_idx] = histc(x_state(1:end-1), edges);
    
    Drift = nan(1, num_bins);
    Diffusion = nan(1, num_bins);
    
    for b = 1:num_bins
        mask = (bin_idx == b);
        % With 30s data, threshold 10 is robust
        if sum(mask) > 10
            Drift(b) = mean(dx(mask)) / dt_sim;
            Diffusion(b) = var(dx(mask)) / dt_sim;
        end
    end
    
    valid = ~isnan(Drift);
    if sum(valid) < 4
        warning('Not enough bins for %s', labels{i});
        results = [results; NaN, NaN];
        potentials = [potentials; nan(1, num_bins)];
        continue;
    end
    
    % 1. Fit Polynomial Force (Stiffness)
    [p, ~] = polyfit(centers(valid), Drift(valid), 3);
    b_stiffness = -p(3)/2;
    
    % 2. Central Temperature (Noise near Equilibrium x=0)
    % Consistent with Clinical Analysis: Fixed window [-1, 1] SD
    center_mask = valid & (centers >= -1) & (centers <= 1);
    
    if sum(center_mask) > 0
        d_temp = mean(Diffusion(center_mask));
    else
        d_temp = mean(Diffusion(valid)); % Fallback to global if center is empty
    end
    
    results = [results; b_stiffness, d_temp];
    
    % 3. Reconstruct Potential for Plotting
    U_curve = -cumsum(Drift(valid)) * (centers(2)-centers(1));
    U_curve = U_curve - min(U_curve);
    
    % Interpolate Potential
    U_interp = interp1(centers(valid), U_curve, x_axis_common, 'linear', NaN);
    potentials = [potentials; U_interp];
end

%% --- 3. Visualization ---
if isempty(results), return; end

fig = figure('Name', 'Mechanistic Comparison (vs Control)', 'Color', 'w', 'Position', [100, 100, 1200, 400]);
t = tiledlayout(1, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% 1. Potential Landscapes
nexttile; hold on;
for i = 1:length(labels)
    plot(x_axis_common, potentials(i,:), 'Color', colors{i}, 'LineWidth', 2.5);
end
xlabel('State (Z)'); ylabel('Potential U(x)');
title('Energy Landscapes'); legend(labels, 'Location', 'north'); grid on;

% 2. Stiffness
nexttile;
b = bar(results(:,1));
b.FaceColor = 'flat';
for i = 1:length(labels), b.CData(i,:) = colors{i}; end
xticklabels(labels); xtickangle(45);
ylabel('Stiffness (b)'); title('Rigidity'); grid on;

% 3. Temperature
nexttile;
b2 = bar(results(:,2));
b2.FaceColor = 'flat';
for i = 1:length(labels), b2.CData(i,:) = colors{i}; end
xticklabels(labels); xtickangle(45);
ylabel('Central Temperature (D)'); title('Noise Level (at Equilibrium)'); grid on;

% Print Interpretation relative to Control
ctrl_stiff = results(1,1);
ctrl_temp = results(1,2);

fprintf('\n--- MECHANISTIC DEVIATIONS ---\n');
for i = 2:length(labels)
    stiff_diff = (results(i,1) - ctrl_stiff) / abs(ctrl_stiff) * 100;
    temp_diff = (results(i,2) - ctrl_temp) / abs(ctrl_temp) * 100;
    fprintf('%s: Stiffness %+.1f%%, Temp %+.1f%%\n', labels{i}, stiff_diff, temp_diff);
end