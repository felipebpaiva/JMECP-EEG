%% SCRIPT 14: Data-Driven RNN Modeling (Ostojic Framework)
% This script fits a Low-Rank Recurrent Neural Network (RNN) to each subject.
%
% THE MODEL: dx/dt = -x + J * tanh(x) + I_noise
% Where J = g_0D * W_0D (Backbone) + g_1D * W_1D (Residual)
%
% GOAL: Find the specific gains (g_0D, g_1D) for each subject such that the
% model's simulated dynamic variability matches the subject's actual 
% EEG smoothness variability.
%
% OUTPUT: The "Spectral Radius" of the fitted J, representing the 
% dynamical regime (Stable vs. Critical vs. Chaotic) of that subject.

clc;
%close all;

%% --- 1. Setup Parameters ---
% Simulation Parameters
dt = 0.5;           % Time step (ms) - Coarse enough for speed, fine enough for stability
T_max = 5000;       % Duration of simulation (ms)
tau = 10;           % Membrane time constant (ms)
noise_amp = 0.5;    % Amplitude of white noise input
n_steps = T_max / dt;
use_gpu = false;    % Set to true ONLY if you have a powerful GPU and 1 worker, otherwise CPU parfor is faster for N=252

% Optimization Grid (The "Search Space" for Gains)
% We search for the best g_0D (Structure) and g_1D (Chaos)
g_range = 0.1:0.2:3.0; 
[G0, G1] = meshgrid(g_range, g_range);

% Pre-allocate Output Matrix for Parfor [WASI, g0, g1, SpecRad, Error]
results_all = nan(nsubj, 5);

%% --- 2. Main Subject Loop (Parallelized) ---
disp('Starting Digital Twin Fitting (Parallelized)...');

% Start parallel pool if not already running
if isempty(gcp('nocreate')), parpool; end

parfor subi = 1:nsubj
    % Skip bad subjects
    if isnan(wasi(subi))
        continue; 
    end
    
    fprintf('Fitting Subject %d...\n', subi);
    
    % --- A. Get Real Data Targets ---
    % 1. Reconstruct Graph & Signal
    % Note: parfor handles slicing of ciPLV_global automatically if indexed by loop var
    A = ciPLV_global(:,:,subi);
    
    % Check for bad matrix inside parfor
    if all(A(:)==0) || any(isnan(A(:)))
        continue; 
    end
    
    % 2. Calculate Laplacian & Eigenvalues (for Smoothness calc)
    D = diag(sum(A, 2));
    L = eye(Nroi) - diag(1./sqrt(diag(D) + eps)) * A * diag(1./sqrt(diag(D) + eps));
    if ~issymmetric(L), L = (L + L') / 2; end
    [V_subj, L_diag] = eig(L);
    [evals_sorted, idx] = sort(diag(L_diag));
    V_subj = V_subj(:, idx);
    
    % 3. Get Actual Smoothness CV (The Target)
    % Handle 4D array slicing for parfor: extract 3D volume first
    elects_sub = all_elects(:,:,:,subi);
    sig_ts = reshape(elects_sub, Nroi, []); 
    
    sig_pow = real(sig_ts).^2;
    gft = V_subj' * sig_pow;
    real_smooth_ts = evals_sorted' * abs(gft).^2;
    
    % TARGET: Coefficient of Variation (CV) 
    target_cv = std(real_smooth_ts) / mean(real_smooth_ts);
    
    % --- B. Construct Structure Matrices (TDA Decomposition) ---
    % 1. W_0D: Backbone (MST)
    % Simplified MST extraction for parfor compatibility
    % Negate A to find Max Spanning Tree using minspantree
    G_neg = graph(-A, 'upper', 'omitselfloops');
    T_mst = minspantree(G_neg); 
    W_0D = abs(adjacency(T_mst, 'weighted')); % Convert back to adjacency
    
    % 2. W_1D: Residual (Full - Backbone)
    W_1D = A - full(W_0D);
    W_1D(W_1D < 0) = 0; % Safety clipping
    
    % Normalize matrices
    rho_0 = max(abs(eig(full(W_0D)))); if rho_0>0, W_0D = W_0D / rho_0; end
    rho_1 = max(abs(eig(full(W_1D)))); if rho_1>0, W_1D = W_1D / rho_1; end
    
    % --- C. Grid Search for Best Fit ---
    best_err = inf;
    best_params = [NaN, NaN];
    best_rho = NaN;
    
    % Pre-generate noise for this subject to use across grid (optional fairness)
    % Generating inside grid loop is also fine
    
    for i = 1:numel(G0)
        g0 = G0(i);
        g1 = G1(i);
        
        % 1. Define J
        J = g0 * W_0D + g1 * W_1D;
        
        % 2. Simulate RNN
        % Initial State
        x = randn(Nroi, 1) * 0.1;
        
        % Pre-generate noise
        I = randn(Nroi, n_steps) * noise_amp;
        
        % GPU Casting (Optional)
        if use_gpu
            J = gpuArray(J);
            x = gpuArray(x);
            I = gpuArray(I);
            X_sim = gpuArray.zeros(Nroi, n_steps);
        else
            X_sim = zeros(Nroi, n_steps);
        end
        
        % Euler Integration Loop
        for t = 1:n_steps
            dx = (-x + J * tanh(x) + I(:,t)) / tau;
            x = x + dx * dt;
            X_sim(:,t) = x.^2; 
        end
        
        if use_gpu
            X_sim = gather(X_sim); % Bring back to CPU for stats
        end
        
        % 3. Calculate Simulated Metric
        X_steady = X_sim(:, round(n_steps*0.2):end); % Discard transient
        
        % Projected Smoothness
        sim_gft = V_subj' * X_steady;
        sim_smooth_ts = evals_sorted' * abs(sim_gft).^2;
        sim_cv = std(sim_smooth_ts) / mean(sim_smooth_ts);
        
        % 4. Compare
        err = abs(sim_cv - target_cv);
        
        if err < best_err
            best_err = err;
            best_params = [g0, g1];
            % Calculate spectral radius only for the best one to save compute
            best_rho = max(abs(eig(full(J)))); % Recalculate on CPU J
        end
    end
    
    % Store in temporary variable
    results_all(subi, :) = [wasi(subi), best_params(1), best_params(2), best_rho, best_err];
end

disp('Modeling Complete. Separating Groups...');

% --- 3. Separate Groups & Plot ---
% Filter out NaNs (skipped subjects)
valid_mask = ~isnan(results_all(:,1));
results_clean = results_all(valid_mask, :);
labels_clean = origLabels(valid_mask);

results_controls = results_clean(labels_clean == 0, :);
results_patients = results_clean(labels_clean == 1, :);

N_extremes = 10;
fig = figure('Name', 'Digital Twin Analysis: Dynamical Regimes', 'WindowState', 'maximized');
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Plot 1: Spectral Radius vs WASI (Controls)
nexttile;
plot_regime_correlation(results_controls, 'Controls', 'b');

% Plot 2: Spectral Radius vs WASI (Patients)
nexttile;
plot_regime_correlation(results_patients, 'Patients', 'r');

% Plot 3: Boxplot of Spectral Radius (Top 10 vs Bottom 10)
nexttile([1,2]); 
compare_extremes(results_controls, results_patients, N_extremes);


%% --- Helper Functions ---
function plot_regime_correlation(data, name, color_code)
    x = data(:,1); % WASI
    y = data(:,4); % Spectral Radius
    
    scatter(x, y, 60, color_code, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    h = lsline; set(h, 'LineWidth', 2, 'Color', 'k');
    yline(1, 'k--', 'Criticality');
    
    [r, p] = corr(x, y, 'Rows', 'complete');
    
    title(sprintf('%s: Regime vs Intelligence\nr=%.2f, p=%.3f', name, r, p));
    xlabel('WASI Score'); ylabel('Spectral Radius (\rho)');
    grid on;
end

function compare_extremes(data_c, data_p, N)
    % Extract Top/Bottom for Controls
    [~, idx_c] = sort(data_c(:,1));
    c_bot = data_c(idx_c(1:N), 4);
    c_top = data_c(idx_c(end-N+1:end), 4);
    
    % Extract Top/Bottom for Patients
    [~, idx_p] = sort(data_p(:,1));
    p_bot = data_p(idx_p(1:N), 4);
    p_top = data_p(idx_p(end-N+1:end), 4);
    
    % Boxplot
    dat = [c_bot; c_top; p_bot; p_top];
    grp = [repmat({'Con-Low'},N,1); repmat({'Con-High'},N,1); ...
           repmat({'Pat-Low'},N,1); repmat({'Pat-High'},N,1)];
       
    boxplot(dat, grp, 'Colors', 'k');
    hold on;
    yline(1, 'k--', 'Criticality Edge');
    ylabel('Spectral Radius (\rho)');
    title('Dynamical Stability of "Digital Twins"');
    grid on;
end