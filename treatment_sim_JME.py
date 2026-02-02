import os
import matplotlib

# --- 1. HEADLESS MODE CONFIGURATION ---
if "DISPLAY" in os.environ:
    del os.environ["DISPLAY"]
os.environ["NRN_NOGUI"] = "1"
matplotlib.use('Agg')

# --- 2. IMPORTS ---
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt, decimate
from hnn_core import jones_2009_model, simulate_dipole

# --- 3. PARAMETER SETS ---

def add_resting_alpha_drive(net):
    """Adds continuous 10 Hz Thalamic drive + Poisson noise."""
    net.add_bursty_drive(
        'alpha_thalamic',
        tstart=0.0,
        burst_rate=10.0,
        burst_std=1.0,
        numspikes=2,
        spike_isi=10.0,
        n_drive_cells=10,
        location='proximal',
        weights_ampa={'L2_pyramidal': 0.0005, 'L5_pyramidal': 0.0005},
        weights_nmda={'L2_pyramidal': 0.0, 'L5_pyramidal': 0.0},
        synaptic_delays=5.0
    )
    net.add_poisson_drive(
        'background_noise',
        tstart=0.0,
        rate_constant={'L2_pyramidal': 10.0, 'L5_pyramidal': 10.0},
        location='distal',
        weights_ampa={'L2_pyramidal': 0.0005, 'L5_pyramidal': 0.0005},
        weights_nmda={'L2_pyramidal': 0.0, 'L5_pyramidal': 0.0}
    )
    return net

def get_disease_model(model_type):
    """Returns a base JME model (GABA, Microdysgenesis, or Arborization)"""
    net = jones_2009_model(add_drives_from_params=False)
    net = add_resting_alpha_drive(net)
    
    if model_type == 'GABA':
        # Deficit: Reduced Inhibition
        for conn in net.connectivity:
            if conn['src_type'] in ['L2_basket', 'L5_basket'] and conn['target_type'] in ['L2_pyramidal', 'L5_pyramidal']:
                conn['nc_dict']['A_weight'] *= 0.6 # 40% reduction
    
    elif model_type == 'Micro':
        # Deficit: Increased Recurrence
        for conn in net.connectivity:
            if conn['src_type'] in ['L2_pyramidal', 'L5_pyramidal'] and conn['target_type'] in ['L2_pyramidal', 'L5_pyramidal']:
                conn['nc_dict']['A_weight'] *= 1.4 # 40% increase

    elif model_type == 'Arbor':
        # Deficit: Reduced Distal Drive
        for conn in net.connectivity:
            if conn['src_type'] == 'background_noise':
                conn['nc_dict']['A_weight'] *= 0.6 # 40% reduction
                
    return net

def apply_treatment(net, drug):
    """
    Simulates the effect of ASM.
    VPA: Increases GABAergic inhibition (Restores or boosts).
    LEV: Reduces vesicle release (Global synaptic gain reduction).
    """
    print(f"   Applying Treatment: {drug}...")
    
    if drug == 'VPA':
        # VPA boosts GABA. We multiply inhibitory weights by 1.5x 
        for conn in net.connectivity:
            if conn['src_type'] in ['L2_basket', 'L5_basket']:
                conn['nc_dict']['A_weight'] *= 1.5 
                
    elif drug == 'LEV':
        # LEV reduces global synaptic gain. We reduce ALL weights.
        for conn in net.connectivity:
            conn['nc_dict']['A_weight'] *= 0.8 # 20% global reduction
            
    return net

# --- 4. SIMULATION RUNNER ---

def run_simulation(net, label):
    print(f"--- Running {label} Simulation (30s) ---")
    dpl = simulate_dipole(net, tstop=30000, dt=0.1, n_trials=1, record_vsec='soma')
    
    raw_signals = []
    trial_data = net.cell_response.vsec[0] 
    
    for cell_type, gid_range in net.gid_ranges.items():
        for gid in gid_range:
            if gid in trial_data and 'soma' in trial_data[gid]:
                raw_signals.append(np.array(trial_data[gid]['soma']))

    X_raw = np.array(raw_signals)
    
    fs_sim = 10000 
    fs_target = 100
    downsample_factor = int(fs_sim / fs_target)
    X_ds = decimate(X_raw, downsample_factor, axis=1)
    
    nyquist = 0.5 * fs_target
    b, a = butter(4, [8 / nyquist, 10 / nyquist], btype='band')
    X_alpha = filtfilt(b, a, X_ds, axis=1)
    
    A_dyn = np.corrcoef(X_alpha)
    np.fill_diagonal(A_dyn, 0)
    A_dyn = np.abs(A_dyn)
    
    filename = f'InSilico_{label}.mat'
    scipy.io.savemat(filename, {'A_dyn': A_dyn, 'elects': X_alpha, 'fs': fs_target})
    print(f"   Saved {filename}")

if __name__ == "__main__":
    
    conditions = ['GABA', 'Micro', 'Arbor']
    treatments = ['Untreated', 'VPA', 'LEV']
    
    for cond in conditions:
        for treat in treatments:
            # 1. Get Base Disease Model
            net = get_disease_model(cond)
            
            # 2. Apply Treatment (if any)
            if treat != 'Untreated':
                net = apply_treatment(net, treat)
                
            # 3. Run
            run_simulation(net, f"{cond}_{treat}")