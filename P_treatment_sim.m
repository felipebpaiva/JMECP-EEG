%% SCRIPT 25: Full In-Silico Pharmacology Grid (Langevin)
% This script applies Langevin Inverse Modeling to the full 3x3 grid of 
% biophysical simulations:
% 3 Conditions: GABA Deficit, Microdysgenesis, Poor Arborization
% 3 Treatments: Untreated, VPA (Boost Inhib), LEV (Reduce Gain)
%
% GOAL: Determine how different mechanisms and treatments shift the 
% thermodynamic state (Stiffness/Rigidity vs. Temperature/Noise).

clc; %close all;

%% --- 1. Setup Parameters ---
dt_sim = 1/100; % Simulation was downsampled to 100 Hz (10ms)
num_bins = 15;  % Robust bin count for simulation data
x_axis_common = linspace(-3, 3, num_bins);

conditions = {'GABA', 'Micro', 'Arbor'};
treatments = {'Untreated', 'VPA', 'LEV'};

% Colors for treatments
% Red (Untreated), Green (VPA), Blue (LEV)
colors = {[0.8 0.2 0.2], [0.2 0.6 0.2], [0.2 0.4 0.8]}; 

% Storage [Conditions x Treatments]
res_stiff = nan(3, 3);
res_temp = nan(3, 3);

%% --- 2. Main Loop ---
disp('Analyzing 3x3 Pharmacology Grid...');

for c = 1:length(conditions)
    for t = 1:length(treatments)
        cond = conditions{c};
        treat = treatments{t};
        
        filename = sprintf('InSilico_%s_%s.mat', cond, treat);
        
        if ~isfile(filename)
            warning('File %s not found. Skipping.', filename);
            continue;
        end
        
        fprintf('Processing %s - %s...\n', cond, treat);
        data = load(filename);
        
        A = double(data.A_dyn);
        X = double(data.elects);
        [N, ~] = size(X);
        
        % --- Smoothness Calculation ---
        A(isnan(A)) = 0;
        D_mat = diag(sum(A, 2));
        d_inv = 1./sqrt(diag(D_mat) + eps);
        L = eye(N) - diag(d_inv) * A * diag(d_inv);
        
        P = X.^2;
        % Global Smoothness Time Series S(t) = x(t)' L x(t)
        smooth_ts = sum(P .* (L * P), 1);
        
        % --- Langevin Reconstruction ---
        x_state = (smooth_ts - mean(smooth_ts)) / std(smooth_ts);
        dx = diff(x_state);
        
        % Adaptive Binning (Robust to outliers)
        limits = prctile(x_state, [2.5, 97.5]);
        % Fallback if flat
        if abs(limits(2)-limits(1)) < 0.1, limits = [-3, 3]; end
        
        edges = linspace(limits(1), limits(2), num_bins+1);
        centers = (edges(1:end-1) + edges(2:end)) / 2;
        
        [~, bin_idx] = histc(x_state(1:end-1), edges);
        
        Drift = nan(1, num_bins);
        Diffusion = nan(1, num_bins);
        
        for b = 1:num_bins
            mask = (bin_idx == b);
            if sum(mask) > 5 % Threshold for valid bin
                Drift(b) = mean(dx(mask)) / dt_sim;
                Diffusion(b) = var(dx(mask)) / dt_sim;
            end
        end
        
        valid = ~isnan(Drift);
        if sum(valid) < 4
            warning('  Not enough valid bins for fit.');
            continue;
        end
        
        % Fit Polynomial Force: F(x) = -ax^3 - bx - c
        [p_poly, ~] = polyfit(centers(valid), Drift(valid), 3);
        
        % Extract Metrics
        b_stiffness = -p_poly(3)/2; % Curvature
        d_temp = mean(Diffusion(valid));
        
        res_stiff(c, t) = b_stiffness;
        res_temp(c, t) = d_temp;
        
        fprintf('  Stiffness=%.3f, Temp=%.3f\n', b_stiffness, d_temp);
    end
end

%% --- 3. Visualization ---
if all(isnan(res_stiff(:))), return; end

fig = figure('Name', 'Pharmacology Grid Results', 'Color', 'w', 'Position', [100, 100, 1000, 600]);
tlay = tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Plot Columns: Conditions
% Plot Rows: Metrics (Stiffness, Temperature)

for c = 1:3
    cond_name = conditions{c};
    
    % --- Row 1: Stiffness ---
    nexttile(c);
    b_plot = bar(1:3, res_stiff(c, :));
    b_plot.FaceColor = 'flat';
    % Color bars by treatment
    for k=1:3, b_plot.CData(k,:) = colors{k}; end
    
    xticks(1:3); xticklabels(treatments); xtickangle(45);
    ylabel('Stiffness (b)');
    title([cond_name ' - Rigidity']);
    grid on;
    
    % --- Row 2: Temperature ---
    nexttile(c+3);
    b_plot2 = bar(1:3, res_temp(c, :));
    b_plot2.FaceColor = 'flat';
    for k=1:3, b_plot2.CData(k,:) = colors{k}; end
    
    xticks(1:3); xticklabels(treatments); xtickangle(45);
    ylabel('Temperature (D)');
    title([cond_name ' - Noise']);
    grid on;
end

% Fake legend for colors
hold on;
h = zeros(3, 1);
for k=1:3
    h(k) = plot(nan, nan, 's', 'MarkerFaceColor', colors{k}, 'MarkerEdgeColor', 'none', 'MarkerSize', 10);
end
lg = legend(h, treatments, 'Orientation', 'horizontal');
lg.Layout.Tile = 'south';