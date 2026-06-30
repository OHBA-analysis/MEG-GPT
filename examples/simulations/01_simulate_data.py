"""Simulate synthetic MEG data."""

# Import packages
import mne
from pathlib import Path
from meg_gpt.data.simulation import TDEBurstSimulation
from meg_gpt.utils.processing import standardize, time_delay_embed
from meg_gpt.utils.post_hoc import functional_connectivity


if __name__ == "__main__":

    # ---------- Setting Up ---------- #

    # Set directories
    BASE_DIR = Path("/well/woolrich/users/olt015/EphysGPT/examples/simulations")
    REAL_DIR = Path("/well/win-camcan/shared/spring23/src")  # real data directory
    SIM_DIR = BASE_DIR / "data_tde"  # simulated data directory

    # Set simulation hyperparameters
    Fs = 100  # sampling frequency
    n_subjects = 10  # number of subjects to simulate
    n_embeddings = 15
    channels_to_use = [10, 20, 30, 40]
    subjects_to_use = [0, 1]
    n_channels = len(channels_to_use)

    print(f"Number of channels: {n_channels}")
    print(f"Number of embeddings: {n_embeddings}")

    # Get TDE covariance matrices from the real data
    tde_covs = []
    for i in subjects_to_use:
        # Get single-subject real data
        data_path = sorted(REAL_DIR.glob(f"*/sflip_parc-raw.fif"))[i]  # select single subject file
        raw = mne.io.read_raw_fif(data_path, preload=True, verbose=False)
        data = raw.get_data(
            picks="misc",
            reject_by_annotation="omit",
            verbose=False,
        ).T  # shape: (n_samples, n_channels)

        original_data = standardize(data, axis=0)
        original_data = original_data[:, channels_to_use]  # select four channels

        # Compute TDE covariance matrix using the real data
        X = time_delay_embed(original_data, n_embeddings)
        X = standardize(X, axis=0)
        tde_cov = functional_connectivity(X, conn_type="cov")
        print("TDE covariance matrix shape: ", tde_cov.shape)

        tde_covs.append(tde_cov)

    print("Number of modes: ", len(tde_covs))

    # ---------- Simulation Configuration ---------- #

    # Set simulation configuration
    simulation_config = {
        "true_tde_covs": tde_covs,
        "n_subjects": n_subjects,
        "n_embeddings": n_embeddings,
        "sampling_frequency": Fs,
        "stay_prob": 0.98,
        "data_dir": SIM_DIR,
    }
    simulation_config["n_samples"] = 5 * 60 * Fs  # simulate 5 minutes of data

    # Initialize simulation
    bursts = TDEBurstSimulation(**simulation_config, rho=1e-9)

    # ---------- Data Simulation ---------- #

    # Simulate and save data
    bursts.simulate(save=True)

    # Plot summary of the simulated data
    bursts.plot_summary(
        idx_start=200,
        idx_end=600,
        plot_dir=(SIM_DIR / "figures"),
        channels_to_plot=[0, 1, 2, 3],
        conn_type="cov",
    )

    print("Simulation complete.")
