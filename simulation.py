# simulation.py


import numpy as np
from config import (
    NEUTRONS_PER_RUN,
    MAX_COLLISIONS,
    SLAB_THICKNESS,
)


def simulate_neutron(sigma_s, sigma_a, sigma_f, rng):
    sigma_t    = sigma_s + sigma_a + sigma_f
    position   = 0.0
    collisions = 0
    scatters   = 0

    p_scatter = sigma_s / sigma_t
    p_absorb  = sigma_a / sigma_t

    while collisions < MAX_COLLISIONS:

        distance  = -np.log(rng.uniform()) / sigma_t
        position += distance

        if SLAB_THICKNESS is not None and position >= SLAB_THICKNESS:
            return 'leakage', collisions, scatters

        collisions += 1
        roll = rng.uniform()

        if roll < p_scatter:
            scatters += 1
            continue
        elif roll < p_scatter + p_absorb:
            return 'absorb', collisions, scatters
        else:
            return 'fission', collisions, scatters

    return 'absorb', collisions, scatters


def run_simulation(sigma_s, sigma_a, sigma_f, n_neutrons=NEUTRONS_PER_RUN, seed=None):
    rng = np.random.default_rng(seed)

    counts = {'absorb': 0, 'fission': 0, 'leakage': 0}
    total_collisions = 0
    total_scatters   = 0

    for _ in range(n_neutrons):
        outcome, collisions, scatters = simulate_neutron(sigma_s, sigma_a, sigma_f, rng)
        counts[outcome] += 1
        total_collisions += collisions
        total_scatters   += scatters

    sigma_t = sigma_s + sigma_a + sigma_f

    return {
        'sigma_s'         : sigma_s,
        'sigma_a'         : sigma_a,
        'sigma_f'         : sigma_f,
        'sigma_t'         : sigma_t,
        'n_neutrons'      : n_neutrons,
        'avg_collisions'  : total_collisions / n_neutrons,
        'avg_scatters'    : total_scatters   / n_neutrons,
        'absorption_frac' : counts['absorb']  / n_neutrons,
        'leakage_frac'    : counts['leakage'] / n_neutrons,
        'fission_rate'    : counts['fission'] / n_neutrons,
    }


# ── Sanity check + Plotly visualisations
if __name__ == "__main__":
    import os
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from config import RESULTS_DIR

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Single run printout
    result = run_simulation(sigma_s=0.5, sigma_a=0.3, sigma_f=0.2, seed=42)

    print("── Simulation Test Run ──────────────────────────────")
    for key, val in result.items():
        print(f"  {key:<25} {val}")
    total = result['fission_rate'] + result['absorption_frac'] + result['leakage_frac']
    print(f"  {'outcome sum':<25} {total:.4f}  (must be 1.0)")
    print("─────────────────────────────────────────────────────")

    # ── Collect per-neutron data
    rng = np.random.default_rng(42)
    collision_data = []
    scatter_data   = []
    outcome_counts = {'absorb': 0, 'fission': 0, 'leakage': 0}

    for _ in range(1000):
        outcome, collisions, scatters = simulate_neutron(0.5, 0.3, 0.2, rng)
        collision_data.append(collisions)
        scatter_data.append(scatters)
        outcome_counts[outcome] += 1

    # Sweep: fission_rate vs sigma_f
    sigma_f_vals  = np.linspace(0.0, 0.4, 40)
    fission_rates = []
    absorb_rates  = []
    leakage_rates = []

    for sf in sigma_f_vals:
        r = run_simulation(sigma_s=0.5, sigma_a=0.3, sigma_f=sf,
                           n_neutrons=1000, seed=42)
        fission_rates.append(r['fission_rate'])
        absorb_rates.append(r['absorption_frac'])
        leakage_rates.append(r['leakage_frac'])

    theoretical = sigma_f_vals / (0.5 + 0.3 + sigma_f_vals)

    #  3D surface: fission_rate over sigma_a x sigma_f grid
    print("Building 3D surface (this takes ~10s)...")
    sa_vals = np.linspace(0.1, 0.6, 25)
    sf_vals = np.linspace(0.0, 0.4, 25)
    Z = np.zeros((len(sa_vals), len(sf_vals)))

    for i, sa in enumerate(sa_vals):
        for j, sf in enumerate(sf_vals):
            if sf > sa:
                Z[i, j] = np.nan
            else:
                r = run_simulation(0.5, sa, sf, n_neutrons=300, seed=42)
                Z[i, j] = r['fission_rate']


    # Plot 1: diagnostics dashboard
    fig1 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Collision Count Distribution",
            "Neutron Outcome Distribution",
            "Fission Rate vs Σ_f",
            "All Outcome Rates vs Σ_f"
        ),
        specs=[
            [{"type": "xy"},     {"type": "domain"}],
            [{"type": "xy"},     {"type": "xy"}    ],
        ]
    )

    # collision histogram
    fig1.add_trace(
        go.Histogram(x=collision_data, nbinsx=30,
                     marker_color='steelblue', opacity=0.85,
                     name='Collisions'),
        row=1, col=1
    )
    fig1.add_vline(
        x=np.mean(collision_data), line_dash='dash', line_color='red',
        annotation_text=f"Mean={np.mean(collision_data):.2f}",
        annotation_position="top right",
        row=1, col=1
    )

    # outcome pie
    fig1.add_trace(
        go.Pie(
            labels=list(outcome_counts.keys()),
            values=list(outcome_counts.values()),
            marker_colors=['#e07b54', '#5b8db8', '#6abf69'],
            hole=0.35, name='Outcomes'
        ),
        row=1, col=2
    )

    # simulated vs theoretical
    fig1.add_trace(
        go.Scatter(x=sigma_f_vals, y=fission_rates,
                   mode='markers', name='Simulated',
                   marker=dict(color='steelblue', size=6)),
        row=2, col=1
    )
    fig1.add_trace(
        go.Scatter(x=sigma_f_vals, y=theoretical,
                   mode='lines', name='Theoretical Σ_f/(Σ_a+Σ_f)',
                   line=dict(color='red', dash='dash')),
        row=2, col=1
    )

    # all outcome rates
    for label, data, color in [
        ('Fission',    fission_rates, '#5b8db8'),
        ('Absorption', absorb_rates,  '#e07b54'),
        ('Leakage',    leakage_rates, '#6abf69'),
    ]:
        fig1.add_trace(
            go.Scatter(x=sigma_f_vals, y=data,
                       mode='lines', name=label,
                       line=dict(color=color)),
            row=2, col=2
        )

    fig1.update_layout(
        title_text="Monte Carlo Neutron Transport — Simulation Diagnostics",
        title_font_size=16,
        height=750,
        template='plotly_dark'
    )
    fig1.update_xaxes(title_text="Collisions",     row=1, col=1)
    fig1.update_yaxes(title_text="Frequency",      row=1, col=1)
    fig1.update_xaxes(title_text="Sigma_f (cm⁻¹)", row=2, col=1)
    fig1.update_yaxes(title_text="Fission Rate",   row=2, col=1)
    fig1.update_xaxes(title_text="Sigma_f (cm⁻¹)", row=2, col=2)
    fig1.update_yaxes(title_text="Rate",           row=2, col=2)

    out1 = os.path.join(RESULTS_DIR, "simulation_diagnostics.html")
    fig1.write_html(out1)
    print(f"Plot 1 saved → {out1}")

    # ══════════════════════════════════════════════════════════════════════════
    # Plot 2: 3D surface
    # ══════════════════════════════════════════════════════════════════════════
    fig2 = go.Figure(data=[
        go.Surface(
            x=sf_vals, y=sa_vals, z=Z,
            colorscale='Plasma',
            colorbar=dict(title='Fission Rate'),
            hovertemplate=(
                "Σ_f: %{x:.3f}<br>"
                "Σ_a: %{y:.3f}<br>"
                "Fission Rate: %{z:.3f}<extra></extra>"
            )
        )
    ])

    fig2.update_layout(
        title='3D Fission Rate Surface — Σ_a vs Σ_f  (Σ_s = 0.5 fixed)',
        scene=dict(
            xaxis_title='Sigma_f (cm⁻¹)',
            yaxis_title='Sigma_a (cm⁻¹)',
            zaxis_title='Fission Rate',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0))
        ),
        template='plotly_dark',
        height=700
    )

    out2 = os.path.join(RESULTS_DIR, "fission_surface_3d.html")
    fig2.write_html(out2)
    print(f"Plot 2 saved → {out2}")

    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(out1)}")
    webbrowser.open(f"file://{os.path.abspath(out2)}")
    print("\nBoth plots opened in browser.")