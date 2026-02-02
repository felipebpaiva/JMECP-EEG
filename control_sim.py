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

def get_control_model():
    """Returns the Standard Jones 2009 model with Resting Alpha drives."""
    # add_drives_from_params=False to ensure no evoked ERP drives are present
    net = jones_2009_model(add_drives_from_params=False)
    net = add_resting_alpha_drive(net)
    return net

# --- 4. SIMULATION RUNNER ---

def run_simulation(net, label):
    print(f"--- Running {label} Simulation (30s) ---")
    
    # Updated tstop to 30000ms (30s) as requested
    dpl = simulate_dipole(net, tstop=30000, dt=0.1, n_trials=1, record_vsec='soma')
    
    raw_signals = []
    # Check for empty vsec (simulation failure check)
    if not net.cell_response.vsec:
        raise RuntimeError("Simulation failed to record somatic voltages.")

    trial_data = net.cell_response.vsec[0] 
    
    for cell_type, gid_range in net.gid_ranges.items():
        for gid in gid_range:
            if gid in trial_data and 'soma' in trial_data[gid]:
                raw_signals.append(np.array(trial_data[gid]['soma']))

    X_raw = np.array(raw_signals)
    print(f"   Extracted {X_raw.shape} neurons.")
    
    # Process
    fs_sim = 10000 # dt=0.1ms -> 10kHz
    fs_target = 100
    downsample_factor = int(fs_sim / fs_target)
    
    # Decimate
    X_ds = decimate(X_raw, downsample_factor, axis=1)
    
    # Alpha Filter
    nyquist = 0.5 * fs_target
    b, a = butter(4, [8 / nyquist, 10 / nyquist], btype='band')
    X_alpha = filtfilt(b, a, X_ds, axis=1)
    
    # Connectivity
    A_dyn = np.corrcoef(X_alpha)
    np.fill_diagonal(A_dyn, 0)
    A_dyn = np.abs(A_dyn)
    
    filename = f'InSilico_{label}.mat'
    scipy.io.savemat(filename, {
        'A_dyn': A_dyn,       
        'elects': X_alpha,    
        'fs': fs_target
    })
    print(f"   Saved {filename}")

if __name__ == "__main__":
    # Run Control Only
    net_ctrl = get_control_model()
    run_simulation(net_ctrl, "Control")