%% SCRIPT 21: Spatially Resolved Langevin Dynamics (Dynamic Graph)
% This script "fine-grains" the Langevin analysis by reconstructing the 
% Potential Landscape for EVERY ELECTRODE individually.
%
% REFINED ANALYSIS: ADAPTIVE BINNING & GRID CLUSTERING
% 1. Uses Time-Resolved Topology (WC_TDA) to calc Nodal Roughness.
% 2. ADAPTIVE BINNING: Uses percentiles (2-98%) to define the range.
% 3. Finds clusters using Z-scored Pixel P-values.
% 4. Extracts cluster averages directly from the grid stack.

clc; %close all;

%% --- 1. Setup Parameters ---
dt = 1/100; 
num_bins = 20; 
z_thresh_cluster = -1.28; 

% Output Matrices [Subjects x Nodes]
if ~exist('curvature_map', 'var')
    curvature_map = nan(nsubj, Nroi);
    run_reconstruction = true;
else
    run_reconstruction = false;
    disp('Using existing curvature_map from workspace.');
end

%% --- 2. Main Loop (Parallelized per Subject) ---
if run_reconstruction
    disp('Starting Spatial Langevin Reconstruction (Dynamic)...');
    if isempty(gcp('nocreate')), parpool; end

    parfor subi = 1:nsubj
        if isnan(wasi(subi)), continue; end
        
        try
            % Load Dynamic Graph [N x N x Trials]
            A_dyn = WC_TDA(:,:,:,subi);
            % Load Signal [N x Time x Trials]
            elects = squeeze(all_elects(:,:,:,subi)); 
            
            [N, ~, n_trials] = size(A_dyn);
            samples_per_trial = size(elects, 2);
            
            % FIX: Calculate total samples correctly
            n_samples_total = samples_per_trial * n_trials;
            
            % Flatten Signal to 2D [N x TotalTime] for easier indexing if needed
            % But we loop trials, so we can keep 3D or reshape carefully
            sig_ts_2d = reshape(elects, N, []); 
        catch
            continue;
        end
        
        % Signal Power
        x_t_all = real(sig_ts_2d).^2;
        
        % Calculate Dynamic Nodal Roughness
        nodal_roughness = nan(N, n_samples_total);
        
        for tr = 1:n_trials
            % Get Graph for this trial
            A = A_dyn(:,:,tr);
            
            % Skip artifact trials (NaNs in graph)
            if any(isnan(A(:))) || all(A(:)==0)
                continue; 
            end
            
            % Calc Laplacian for this trial
            D = diag(sum(A, 2));
            L = eye(N) - diag(1./sqrt(diag(D) + eps)) * A * diag(1./sqrt(diag(D) + eps));
            
            % Get Signal indices for this trial
            idx_start = (tr-1)*samples_per_trial + 1;
            idx_end = tr*samples_per_trial;
            
            x_chunk = x_t_all(:, idx_start:idx_end);
            
            % Calc Roughness: r = x .* (L * x)
            Lx = L * x_chunk;
            nodal_roughness(:, idx_start:idx_end) = x_chunk .* Lx;
        end
        
        % --- Langevin Reconstruction ---
        sub_curve = nan(1, N);
        
        for n = 1:N
            ts = nodal_roughness(n, :);
            ts = ts(~isnan(ts)); 
            
            % NOW this check should pass (ts length ~18000)
            if length(ts) < 1000, continue; end 
            
            ts_z = (ts - mean(ts)) / (std(ts) + eps);
            dx = diff(ts_z);
            
            % Adaptive Range
            limits = prctile(ts_z, [2.5, 97.5]);
            if abs(limits(2)-limits(1)) < 1e-6, limits = [-3, 3]; end
            
            edges = linspace(limits(1), limits(2), num_bins+1);
            centers = (edges(1:end-1) + edges(2:end)) / 2;
            
            Drift = nan(1, num_bins);
            [~, bin_idx] = histc(ts_z(1:end-1), edges);
            
            for b = 1:num_bins
                mask = (bin_idx == b);
                if sum(mask) > 10 
                    Drift(b) = mean(dx(mask)) / dt;
                end
            end
            
            valid = ~isnan(Drift);
            if sum(valid) >= 4
                [p, ~] = polyfit(centers(valid), Drift(valid), 3);
                sub_curve(n) = -p(3)/2; 
            end
        end
        curvature_map(subi, :) = sub_curve;
    end
    
    success_rate = sum(~isnan(curvature_map(:))) / numel(curvature_map);
    fprintf('Reconstruction Complete. Success Rate: %.1f%%\n', success_rate*100);
end

%% --- 3. Grid Projection (Pre-Calculate All Subjects) ---
disp('Projecting all subjects to Grid (Grid-First Approach)...');
valid_mask = ~isnan(wasi);
curv_clean = curvature_map(valid_mask, :);
labels_clean = origLabels(valid_mask);
wasi_clean = wasi(valid_mask);

idx_c = labels_clean == 0;
idx_p = labels_clean == 1;

dummy_vals = zeros(1, size(curv_clean, 2));
[~, ~, ~, Xi, Yi] = topoplot(dummy_vals, EEG.chanlocs, 'noplot', 'on', 'gridscale', 67, 'plotrad', 0.5);
grid_sz = size(Xi);

grids_c = project_group_to_grid(curv_clean(idx_c, :), EEG.chanlocs, grid_sz);
grids_p = project_group_to_grid(curv_clean(idx_p, :), EEG.chanlocs, grid_sz);

%% --- 4. Pixel-wise Robust Statistics ---
disp('Calculating Pixel-wise Robust Regressions...');
stats_c = calc_pixel_stats_robust(grids_c, wasi_clean(idx_c));
stats_p = calc_pixel_stats_robust(grids_p, wasi_clean(idx_p));

%% --- 5. Grid Clustering ---
cluster_c = find_grid_clusters_morph(stats_c, z_thresh_cluster, 'Controls', Xi, Yi);
cluster_p = find_grid_clusters_morph(stats_p, z_thresh_cluster, 'Patients', Xi, Yi);

%% --- 6. Plotting ---
fig = figure('Name', 'Clustered Stiffness (Morphologically Closed)', 'WindowState', 'maximized');
t = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% --- CONTROLS ---
nexttile;
plot_grid_topo_direct(cluster_c, stats_c.t, EEG.chanlocs, 'Controls: Robust T-Map');

nexttile;
plot_grid_scatter_direct(grids_c, wasi_clean(idx_c), cluster_c, 'Controls', 'b');

% --- PATIENTS ---
nexttile;
plot_grid_topo_direct(cluster_p, stats_p.t, EEG.chanlocs, 'Patients: Robust T-Map');

nexttile;
plot_grid_scatter_direct(grids_p, wasi_clean(idx_p), cluster_p, 'Patients', 'r');


%% --- Helper Functions ---

function grid_stack = project_group_to_grid(data, locs, g_size)
    n_sub = size(data, 1);
    grid_stack = zeros(n_sub, g_size(1), g_size(2));
    
    f_dummy = figure('Visible', 'off');
    for i = 1:n_sub
        vals = data(i,:);
        if any(isnan(vals))
            vals(isnan(vals)) = mean(vals, 'omitnan');
        end
        
        try
            [~, Zi] = topoplot(vals, locs, 'noplot', 'on', 'gridscale', 67, 'plotrad', 0.5);
            grid_stack(i, :, :) = Zi;
        catch
            grid_stack(i, :, :) = NaN;
        end
    end
    close(f_dummy);
end

function stats = calc_pixel_stats_robust(grid_stack, scores)
    [~, dim1, dim2] = size(grid_stack);
    stats.t = nan(dim1, dim2);
    stats.p = nan(dim1, dim2);
    
    x = (scores - mean(scores)) / std(scores); 
    
    for i = 1:dim1
        for j = 1:dim2
            y_raw = grid_stack(:, i, j);
            if any(isnan(y_raw)), continue; end
            y = (y_raw - mean(y_raw)) / std(y_raw); 
            
            try
                [~, s] = robustfit(x, y);
                stats.p(i,j) = s.p(2);
                stats.t(i,j) = s.t(2);
            catch
                % Keep NaNs
            end
        end
    end
end

function clust = find_grid_clusters_morph(stats, z_thresh, label, Xi, Yi)
    grid_p = stats.p;
    grid_t = stats.t;
    clust.Xi = Xi; clust.Yi = Yi;
    
    p_safe = grid_p; 
    p_safe(isnan(p_safe)) = 1;
    grid_z_stat = norminv(p_safe);
    
    binary_map = grid_z_stat < z_thresh;
    
    se = strel('disk', 1); 
    binary_map = imclose(binary_map, se);
    
    CC = bwconncomp(binary_map);
    numPixels = cellfun(@numel, CC.PixelIdxList);
    
    fprintf('[%s] Found %d clusters.\n', label, CC.NumObjects);
    
    if isempty(numPixels)
        clust.mask = false(size(grid_p));
        clust.isEmpty = true;
        return;
    end
    
    size_thresh = prctile(numPixels, 95); % change size here
    valid_idx = find(numPixels >= size_thresh);
    
    if isempty(valid_idx)
        clust.mask = false(size(grid_p));
        clust.isEmpty = true;
        return;
    end
    
    best_clust_id = 0;
    max_score = -inf;
    
    for k = valid_idx
        pixels = CC.PixelIdxList{k};
        curr_score = max(abs(mean(grid_t(pixels)))); 
        if curr_score > max_score
            max_score = curr_score;
            best_clust_id = k;
        end
    end
    
    mask = false(size(grid_p));
    mask(CC.PixelIdxList{best_clust_id}) = true;
    
    clust.mask = mask;
    clust.isEmpty = false;
end

function plot_grid_topo_direct(clust, t_map, locs, title_str)
    topoplot(zeros(1, length(locs)), locs, 'plotrad', 0.5, 'style', 'blank', 'electrodes', 'off'); 
    hold on;
    
    if clust.isEmpty
        title([title_str ' (None)']);
        return;
    end
    
    masked_t = t_map;
    masked_t(~clust.mask) = NaN; 
    
    surf(clust.Xi, clust.Yi, zeros(size(masked_t))+0.1, masked_t, 'EdgeColor', 'none', 'FaceColor', 'interp');
    view(2); colormap('jet'); caxis([-3 3]); 
    cb = colorbar; cb.Label.String = 'Robust T-Stat';
    title(title_str);
end

function plot_grid_scatter_direct(grid_stack, scores, clust, label, color_c)
    if clust.isEmpty
        text(0.5, 0.5, 'No Clusters', 'HorizontalAlignment','center');
        return;
    end
    
    n_subs = size(grid_stack, 1);
    cluster_avgs = zeros(n_subs, 1);
    mask = clust.mask;
    
    for i = 1:n_subs
        img = squeeze(grid_stack(i, :, :));
        cluster_avgs(i) = mean(img(mask), 'omitnan');
    end
    
    y = (cluster_avgs - mean(cluster_avgs)) / std(cluster_avgs);
    x = scores;
    
    scatter(x, y, 60, color_c, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    
    [b, stats] = robustfit((x-mean(x))/std(x), y); 
    
    [b_plot, ~] = robustfit(x, y);
    x_grid = linspace(min(x), max(x), 100)';
    plot(x_grid, b_plot(1) + b_plot(2)*x_grid, 'Color', color_c, 'LineWidth', 2);
    
    title(sprintf('Cluster Stiffness vs WASI\nRobust Beta=%.2f, p=%.3f', b(2), stats.p(2)));
    xlabel('WASI'); ylabel('Cluster Mean Stiffness (Z)');
    grid on;
end