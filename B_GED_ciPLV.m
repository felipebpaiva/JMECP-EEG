%% SCRIPT 1: GED, Time-Series Reconstruction, and Graph Calculation
% This script takes raw, preprocessed EEG data and performs the following:
% 1. Applies Generalized Eigendecomposition (GED) to isolate alpha-band activity.
% 2. Reconstructs a clean alpha-band time series for each subject based on the original methodology.
% 3. Calculates dynamic (trial-by-trial) and global functional connectivity
%    graphs using the corrected imaginary Phase Locking Value (ciPLV).
% 4. Saves all generated data into a single .mat file for subsequent analysis.
%
% **NOTE**: This version is parallelized using parfor for significant speedup.

clear all;
clc;

%% Load EEGLAB
eeglab;

%% --- 1. Setup and Parameters ---
disp('Setting up parameters...');
ltrials = 5;       % length of trials in seconds
nsec = 180;        % total number of seconds to be analyzed
ntrials = floor(nsec/ltrials);
resfreq = 100;     % frequency for resampling
freq = [8,12];     % alpha-band frequency range
Nroi = 252;        % number of electrodes
sub2ana = 1:142;   % subjects to analyze
condit = 'EC';     % eyes closed
sec2jump = 15;     % seconds to ignore at the beginning of the recording
sectrials = ntrials*ltrials;

% Load demographic and identifying information
full_table = readtable(['K:\JMECP_EEG_Analysis\JMECP_EEG_dmgrphcs_' condit '.xlsx']);
origLabels = table2array(full_table(1:length(sub2ana),6));
baddata = table2array(full_table(1:length(sub2ana),7));
subcode = table2array(full_table(1:length(sub2ana),10));

% Initialize matrices for all possible subjects
WC_TDA = nan(Nroi, Nroi, ntrials, length(sub2ana));
ciPLV_global = nan(Nroi, Nroi, length(sub2ana));
alphamap = nan(length(sub2ana), Nroi);
gamma = nan(length(sub2ana), 2);
selec_comps = nan(1, length(sub2ana));
alphaPower = nan(ntrials, Nroi, length(sub2ana));
subjects_to_keep = false(1, length(sub2ana)); % Logical index for valid subjects

% --- Start Parallel Pool ---
if isempty(gcp('nocreate'))
    parpool;
end
disp('Parallel pool started.');


%% --- 2. Main Data Processing Loop (Parallelized) ---
% This loop processes each subject independently.
parfor subi = sub2ana
    subj = ['sub-' num2str(subi,'%03d')];
    fprintf('Processing %s \n',subj)
    
    file2open = ['K:\JMECP_EEG_Analysis\JMECP_BIDS\RELAX_prepro_EC_v2\RELAXProcessed\Cleaned_Data\' subj '_rs-EC_eegRAW_RELAX.set'];
    dir2open = dir(file2open);

    if (~isempty(dir2open) && baddata(subi) ~= 1)
        EEG = pop_loadset(file2open);
        EEG = pop_resample(EEG, resfreq);
        EEG = eeg_checkset(EEG);

        % Filter data into the alpha band
        EEGalpha = pop_eegfiltnew(EEG, freq(1), freq(2));
        EEGalpha = eeg_checkset(EEGalpha);
        alphafilt = EEGalpha.data;

        % Segment into trials
        jump = sec2jump * EEG.srate;
        Rdata = reshape(EEG.data(:, jump+1:sectrials*EEG.srate+jump), EEG.nbchan, [], ntrials);
        Sdata = reshape(alphafilt(:, jump+1:sectrials*EEG.srate+jump), EEG.nbchan, [], ntrials);

        % Reshape and mean-center data for covariance calculation
        Rdata_m = reshape(Rdata - mean(Rdata, 2), Nroi, []);
        Sdata_m = reshape(Sdata - mean(Sdata, 2), Nroi, []);

        % Ledoit-Wolf shrinkage for robust covariance estimation
        [covR, gamma_R] = ledoit_wolf(Rdata_m');
        [covS, gamma_S] = ledoit_wolf(Sdata_m');
        
        % Generalized Eigendecomposition (GED)
        [evecs, evals] = eig(covR, covS);
        [evals, sidx] = sort(diag(evals), 'ascend');
        evecs = evecs(:, sidx);
        
        % Select number of components
        n_components_found = find(diff(evals) < (0.05 * evals(1:end-1)), 1);
        min_comps = 3; max_comps = 12;
        if isempty(n_components_found) || n_components_found < min_comps
            n_components = min_comps;
        else
            n_components = n_components_found;
        end
        n_components = min(n_components, max_comps);
        
        elects = zeros(Nroi, size(Rdata_m,2));
        avgmap = zeros(1, Nroi);
        evals_sel = evals(1:n_components);
        evecs_sel = evecs(:,1:n_components);
        
        for comp2ana = 1:n_components
            % **MODIFIED**: Use alpha-filtered data for component time series
            compts = evecs_sel(:,comp2ana)' * Rdata_m;
            
            compmap = evecs_sel(:,comp2ana)' * covS;
            
            % **MODIFIED**: Polarity correction now flips both map and time series
            [~,idx] = max(abs(compmap));
            if compmap(idx)<0
                compmap = compmap * -1;
                compts = compts * -1; % Flip time series as well
            end
            
            compmap = (compmap-min(compmap)) / (max(compmap)-min(compmap));
            elects = elects + ((compmap' * compts) * (1/evals_sel(comp2ana)));
            avgmap = avgmap + (compmap * (1/evals_sel(comp2ana)));
        end
        
        elects = reshape(elects, Nroi, [], ntrials);
        avgmap = zscore(avgmap / n_components);

        % Calculate ciPLV for each trial and globally
        complex_plv_global = nan(Nroi, Nroi, ntrials);
        ciPLV_trial = nan(Nroi, Nroi, ntrials);
        alphaP_trial = nan(ntrials, Nroi);
        
        for trialidx = 1:ntrials
            trial_data = squeeze(elects(:,:,trialidx));
            alphaP_trial(trialidx, :) = bandpower(trial_data', resfreq, freq);
            
            as_global = hilbert(trial_data')';
            nas_global = as_global ./ abs(as_global);
            complex_plv_global(:,:,trialidx) = (nas_global * nas_global') / size(nas_global, 2);
            
            window_size = floor(size(trial_data, 2) / 10);
            complex_plv_windows = nan(Nroi, Nroi, 10);
            for window = 1:10
                start_idx = (window-1) * window_size + 1;
                end_idx = window * window_size;
                as_win = hilbert(trial_data(:, start_idx:end_idx)')';
                nas_win = as_win ./ abs(as_win);
                complex_plv_windows(:,:,window) = (nas_win * nas_win') / size(nas_win, 2);
            end
            
            avg_complex_plv = mean(complex_plv_windows, 3);
            imag_part = imag(avg_complex_plv);
            real_part = real(avg_complex_plv);
            real_part(real_part > 1) = 1; real_part(real_part < -1) = -1;
            denominator = sqrt(1 - real_part.^2);
            ciPLV = imag_part ./ denominator;
            ciPLV(denominator == 0) = 0;
            ciPLV_trial(:,:,trialidx) = ciPLV;
        end
        
        % Global ciPLV (averaged across all trials)
        avg_complex_plv_global = mean(complex_plv_global, 3);
        imag_part_g = imag(avg_complex_plv_global);
        real_part_g = real(avg_complex_plv_global);
        real_part_g(real_part_g > 1) = 1; real_part_g(real_part_g < -1) = -1;
        denominator_g = sqrt(1 - real_part_g.^2);
        ciPLV_g = imag_part_g ./ denominator_g;
        ciPLV_g(denominator_g == 0) = 0;
        
        % Assign results to pre-allocated matrices
        ciPLV_global(:,:,subi) = ciPLV_g;
        WC_TDA(:,:,:,subi) = ciPLV_trial;
        alphamap(subi,:) = avgmap;
        alphaPower(:,:,subi) = alphaP_trial;
        gamma(subi,:) = [gamma_R, gamma_S];
        selec_comps(subi) = n_components;
        
        subjects_to_keep(subi) = true;
    else
        fprintf('Skipping subject %d due to missing data or bad data flag.\n', subi);
    end
end

%% --- 3. Clean and Save Data ---
disp('Finalizing and saving data...');

origLabels = origLabels(subjects_to_keep);
subcode = subcode(subjects_to_keep);
ciPLV_global = ciPLV_global(:,:,subjects_to_keep);
WC_TDA = WC_TDA(:,:,:,subjects_to_keep);
alphamap = alphamap(subjects_to_keep,:);
alphaPower = alphaPower(:,:,subjects_to_keep);
gamma = gamma(subjects_to_keep,:);
selec_comps = selec_comps(subjects_to_keep);
nsubj = length(origLabels);

save('K:\JMECP_EEG_Analysis\results\RELAXv2\Preprocessed_GED_Graphs_5s_nofilt.mat', ...
    'WC_TDA', 'ciPLV_global', 'alphamap', 'alphaPower', 'gamma', 'selec_comps', ...
    'nsubj', 'origLabels', 'subcode', 'Nroi', 'ntrials', 'EEG', 'baddata', 'ltrials', '-v7.3');

disp('Preprocessing complete. Data saved.');


%% --- Helper Functions ---
function [Sigma_LW, gamma] = ledoit_wolf(X)
    [N, p] = size(X);
    S = cov(X);
    mu = trace(S) / p;
    F = mu * eye(p);
    d2 = norm(S - F, 'fro')^2;
    Xc = X - mean(X, 1);
    b2_sum = 0;
    for i = 1:N
        outer_product = Xc(i, :)' * Xc(i, :);
        b2_sum = b2_sum + norm(outer_product - S, 'fro')^2;
    end
    b2 = b2_sum / N^2;
    if d2 == 0, gamma = 0; else, gamma = b2 / d2; end
    gamma = max(0, min(1, gamma));
    Sigma_LW = (1 - gamma) * S + gamma * F;
end

