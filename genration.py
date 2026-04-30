
import numpy as np
import pandas as pd
import os
import time
from config import (
    NUM_RUNS,
    NEUTRONS_PER_RUN,
    SIGMA_S_RANGE,
    SIGMA_A_RANGE,
    SIGMA_F_RANGE,
    RANDOM_SEED,
    DATASET_PATH,
    RESULTS_DIR,
)
from simulation import run_simulation


def generate_dataset(num_runs=NUM_RUNS, n_neutrons=NEUTRONS_PER_RUN, seed=RANDOM_SEED):
    """
    Generate a dataset of Monte Carlo simulation runs.

    Each row = one simulation run with a unique material configuration.
    Cross-sections are sampled uniformly from ranges in config.py.

    Physics constraint enforced:
        sigma_f <= sigma_a   (fission is a subset of absorption)

    Parameters
    ----------
    num_runs   : int — number of rows to generate
    n_neutrons : int — neutrons simulated per run
    seed       : int — master random seed

    Returns
    -------
    pd.DataFrame
    """

    rng  = np.random.default_rng(seed)
    rows = []

    print(f"Generating {num_runs:,} simulation runs ({n_neutrons:,} neutrons each)...")
    print("─" * 55)

    start = time.time()

    for i in range(num_runs):

        # ── Sample cross-sections
        sigma_s = rng.uniform(*SIGMA_S_RANGE)
        sigma_a = rng.uniform(*SIGMA_A_RANGE)

        # enforce: sigma_f <= sigma_a (physics constraint)
        sigma_f_max = min(SIGMA_F_RANGE[1], sigma_a)
        sigma_f     = rng.uniform(SIGMA_F_RANGE[0], sigma_f_max)

        # ── Run simulation
        # pass a unique seed per run derived from master rng
        run_seed = int(rng.integers(0, 2**31))
        result   = run_simulation(sigma_s, sigma_a, sigma_f,
                                  n_neutrons=n_neutrons, seed=run_seed)
        rows.append(result)

        # ── Progress indicator
        if (i + 1) % (num_runs // 10) == 0:
            pct     = (i + 1) / num_runs * 100
            elapsed = time.time() - start
            eta     = elapsed / (i + 1) * (num_runs - i - 1)
            print(f"  {pct:5.1f}%  |  {i+1:>6,} rows  |  "
                  f"elapsed {elapsed:.1f}s  |  ETA {eta:.1f}s")

    elapsed = time.time() - start
    print("─" * 55)
    print(f"Done. {num_runs:,} rows generated in {elapsed:.1f}s\n")

    return pd.DataFrame(rows)


def validate_dataset(df):
    """
    Run basic sanity checks on the generated dataset.
    Prints a report does not modify the dataframe.
    """

    print("── Dataset Validation ───────────────────────────────")

    # ── Shape ─────────────────────────────────────────────────
    print(f"  Shape              : {df.shape}")

    # ── Missing values ─────────────────────────────────────────
    nulls = df.isnull().sum().sum()
    print(f"  Missing values     : {nulls}  {'✓' if nulls == 0 else '✗ PROBLEM'}")

    # ── Outcome sum check ──────────────────────────────────────
    outcome_sum = df['fission_rate'] + df['absorption_frac'] + df['leakage_frac']
    max_err     = (outcome_sum - 1.0).abs().max()
    print(f"  Max outcome error  : {max_err:.6f}  {'✓' if max_err < 0.01 else '✗ PROBLEM'}")

    # ── Physics constraint: sigma_f <= sigma_a ─────────────────
    violated = (df['sigma_f'] > df['sigma_a']).sum()
    print(f"  sigma_f > sigma_a  : {violated} violations  {'✓' if violated == 0 else '✗ PROBLEM'}")

    # ── Target range ──────────────────────────────────────────
    print(f"  fission_rate range : [{df['fission_rate'].min():.4f}, "
          f"{df['fission_rate'].max():.4f}]")
    print(f"  fission_rate mean  : {df['fission_rate'].mean():.4f}")

    # ── Feature stats ──────────────────────────────────────────
    print(f"\n  Feature summary:")
    print(df[['sigma_s', 'sigma_a', 'sigma_f', 'sigma_t',
              'avg_collisions', 'fission_rate']].describe().round(4).to_string())

    print("─" * 55)


def save_dataset(df, path=DATASET_PATH):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Dataset saved → {path}  ({os.path.getsize(path)/1024:.1f} KB)")


def plot_dataset(df):
    """
    Four diagnostic plots of the generated dataset.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dataset Diagnostics — Monte Carlo Neutron Transport", fontsize=14)

    # ── Plot 1: fission_rate distribution ─────────────────────
    axes[0, 0].hist(df['fission_rate'], bins=50, color='steelblue', edgecolor='white')
    axes[0, 0].set_title("Target Distribution: Fission Rate")
    axes[0, 0].set_xlabel("fission_rate")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].axvline(df['fission_rate'].mean(), color='red', linestyle='--',
                       label=f"Mean = {df['fission_rate'].mean():.3f}")
    axes[0, 0].legend()

    # ── Plot 2: sigma_f vs fission_rate scatter ────────────────
    axes[0, 1].scatter(df['sigma_f'], df['fission_rate'],
                       alpha=0.3, s=5, color='steelblue')
    axes[0, 1].set_title("Sigma_f vs Fission Rate")
    axes[0, 1].set_xlabel("sigma_f")
    axes[0, 1].set_ylabel("fission_rate")

    # ── Plot 3: sigma_t vs avg_collisions ─────────────────────
    axes[1, 0].scatter(df['sigma_t'], df['avg_collisions'],
                       alpha=0.3, s=5, color='#e07b54')
    axes[1, 0].set_title("Sigma_t vs Avg Collisions")
    axes[1, 0].set_xlabel("sigma_t")
    axes[1, 0].set_ylabel("avg_collisions")

    # ── Plot 4: correlation heatmap ───────────────────────────
    cols = ['sigma_s', 'sigma_a', 'sigma_f', 'sigma_t',
            'avg_collisions', 'absorption_frac', 'leakage_frac', 'fission_rate']
    corr = df[cols].corr()
    im   = axes[1, 1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(cols)))
    axes[1, 1].set_yticks(range(len(cols)))
    axes[1, 1].set_xticklabels(cols, rotation=45, ha='right', fontsize=8)
    axes[1, 1].set_yticklabels(cols, fontsize=8)
    axes[1, 1].set_title("Feature Correlation Heatmap")
    plt.colorbar(im, ax=axes[1, 1])

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "dataset_diagnostics.png")
    plt.savefig(out, dpi=150)
    print(f"Plot saved → {out}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = generate_dataset()
    validate_dataset(df)
    save_dataset(df)
    plot_dataset(df)