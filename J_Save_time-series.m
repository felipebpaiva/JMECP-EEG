%% SCRIPT 1a: Save Reconstructed Time Series
% This script takes raw, preprocessed EEG data, performs GED, reconstructs
% the alpha-band time series using the agreed-upon method (filtered data projection,
% map+time series polarity correction), and saves the resulting time series.
% This output is needed for subsequent GSP-based SDI analysis.
%
% **NOTE**: This version is parallelized using parfor.

clear all;
clc;

%% Load EEGLAB
eeglab;

%% --- 1. Setup and Parameters ---
disp('Setting up parameters...');
ltrials = 3;       % length of trials in seconds
nsec = 180;        % total number of seconds to be analyzed
ntrials = floor(nsec/ltrials);
resfreq = 100;     % frequency for resampling
freq = [8,10];     % alpha-band frequency range
Nroi = 252;        % number of electrodes
sub2ana = 1:142;   % subjects to analyze
condit = 'EC';     % eyes closed
sec2jump = 15;     % seconds to ignore at the beginning of the recording
sectrials = ntrials*ltrials;
samples_per_trial = ltrials * resfreq; % Calculate samples per trial

% Load demographic and identifying information
full_table = readtable(['K:\JMECP_EEG_Analysis\JMECP_EEG_dmgrphcs_' condit '.xlsx']);
origLabels = table2array(full_table(1:length(sub2ana),6));
baddata = table2array(full_table(1:length(sub2ana),7));
subcode = table2array(full_table(1:length(sub2ana),10));

% Initialize matrices for all possible subjects
% **FIX**: Corrected dimensions (nodes x samples x trials x subjects)
all_elects = nan(Nroi, samples_per_trial, ntrials, length(sub2ana)); 
alphamap = nan(length(sub2ana), Nroi); % Keep for potential QC
gamma = nan(length(sub2ana), 2); % Keep for potential QC
selec_comps = nan(1, length(sub2ana)); % Keep for potential QC
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
        EEG_load = pop_loadset(file2open); % Use temporary variable inside parfor
        EEG_load = pop_resample(EEG_load, resfreq);
        EEG_load = eeg_checkset(EEG_load);

        % Filter data into the alpha band
        EEGalpha = pop_eegfiltnew(EEG_load, freq(1), freq(2));
        EEGalpha = eeg_checkset(EEGalpha);
        alphafilt = EEGalpha.data;

        % Segment into trials
        jump = sec2jump * EEG_load.srate;
        Rdata = reshape(EEG_load.data(:, jump+1:sectrials*EEG_load.srate+jump), EEG_load.nbchan, [], ntrials);
        Sdata = reshape(alphafilt(:, jump+1:sectrials*EEG_load.srate+jump), EEG_load.nbchan, [], ntrials);

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
        
        elects_subj = zeros(Nroi, size(Sdata_m,2)); % Temporary variable for this subject
        avgmap_subj = zeros(1, Nroi); % Temporary variable
        evals_sel = evals(1:n_components);
        evecs_sel = evecs(:,1:n_components);
        
        for comp2ana = 1:n_components
            compts = evecs_sel(:,comp2ana)' * Sdata_m;
            compmap = evecs_sel(:,comp2ana)' * covS;
            [~,idx] = max(abs(compmap));
            if compmap(idx)<0
                compmap = compmap * -1;
                compts = compts * -1; % Flip time series as well
            end
            compmap = (compmap-min(compmap)) / (max(compmap)-min(compmap));
            elects_subj = elects_subj + ((compmap' * compts) * (1/evals_sel(comp2ana)));
            avgmap_subj = avgmap_subj + (compmap * (1/evals_sel(comp2ana)));
        end
        
        % Reshape to nodes x samples x trials for this subject
        elects_reshaped = reshape(elects_subj, Nroi, [], ntrials); 
        
        % **FIX**: Correct assignment using 4th dimension for subject
        all_elects(:,:,:,subi) = elects_reshaped; 
        
        avgmap = zscore(avgmap_subj / n_components);
        
        % Assign other QC metrics (use temporary variables if needed)
        alphamap(subi,:) = avgmap; 
        gamma(subi,:) = [gamma_R, gamma_S];
        selec_comps(subi) = n_components;
        
        subjects_to_keep(subi) = true;
    else
        fprintf('Skipping subject %d due to missing data or bad data flag.\n', subi);
    end
end

%% --- 3. Clean and Save Data ---
disp('Finalizing and saving time series data...');

% Use the logical index to remove unprocessed subjects
origLabels = origLabels(subjects_to_keep);
subcode = subcode(subjects_to_keep);
% **FIX**: Correct slicing for the 4D matrix
all_elects = all_elects(:,:,:,subjects_to_keep); 
alphamap = alphamap(subjects_to_keep,:);
gamma = gamma(subjects_to_keep,:);
selec_comps = selec_comps(subjects_to_keep);
nsubj = length(origLabels); % Update the final subject count

% Save only the necessary variables for the next step
save('K:\JMECP_EEG_Analysis\results\RELAXv2\Preprocessed_GED_timeseries_8to10hz.mat', ...
    'all_elects', '-v7.3'); % Keep EEG for chanlocs

disp('Reconstructed time series saved.');


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
