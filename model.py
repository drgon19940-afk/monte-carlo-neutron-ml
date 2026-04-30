import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics         import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing   import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import DATASET_PATH, RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)


# 1. Load & split

print("── Loading dataset ───────────────────────────────────")
df = pd.read_csv(DATASET_PATH)
print(f"  Shape   : {df.shape}")
print(f"  Target  : fission_rate  "
      f"[{df['fission_rate'].min():.4f}, {df['fission_rate'].max():.4f}]")

FEATURES = ['sigma_s', 'sigma_a', 'sigma_f', 'sigma_t',
            'avg_collisions', 'avg_scatters']
TARGET   = 'fission_rate'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train   : {len(X_train):,} rows")
print(f"  Test    : {len(X_test):,} rows\n")

scaler   = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test  = scaler.transform(X_test)


# 2. Train models

print("── Training models ───────────────────────────────────")

lr = LinearRegression()
lr.fit(Xs_train, y_train)
print("  Linear Regression     ✓")

rf = RandomForestRegressor(
    n_estimators=100, max_depth=None,
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
print("  Random Forest         ✓")

gb = GradientBoostingRegressor(
    n_estimators=100, learning_rate=0.1,
    max_depth=4, subsample=0.8, random_state=42
)
gb.fit(X_train, y_train)
print("  Gradient Boosting     ✓\n")


# 3. Evaluate on test set

def evaluate(name, model, X_t, y_t):
    pred = model.predict(X_t)
    return {
        'name' : name,
        'pred' : pred,
        'r2'   : r2_score(y_t, pred),
        'rmse' : np.sqrt(mean_squared_error(y_t, pred)),
        'mae'  : mean_absolute_error(y_t, pred),
        'resid': y_t.values - pred,
    }

lr_res = evaluate("Linear Regression", lr, Xs_test, y_test)
rf_res = evaluate("Random Forest",     rf, X_test,  y_test)
gb_res = evaluate("Gradient Boosting", gb, X_test,  y_test)
results = [lr_res, rf_res, gb_res]

print("── Test Set Results ──────────────────────────────────")
print(f"  {'Model':<22} {'R²':>7}  {'RMSE':>7}  {'MAE':>7}")
print("  " + "─" * 45)
for r in results:
    print(f"  {r['name']:<22} {r['r2']:>7.4f}  {r['rmse']:>7.4f}  {r['mae']:>7.4f}")
print()


# 4. Cross-validation test (5-fold)

print("── Cross-Validation (5-fold R²) ──────────────────────")
cv_configs = [
    ("Linear Regression", lr, Xs_train),
    ("Random Forest",     rf, X_train),
    ("Gradient Boosting", gb, X_train),
]
cv_scores = {}
for name, model, Xt in cv_configs:
    scores = cross_val_score(model, Xt, y_train, cv=5, scoring='r2', n_jobs=-1)
    cv_scores[name] = scores
    print(f"  {name:<22}  mean={scores.mean():.4f}  std={scores.std():.4f}")
print()


# 5. Overfitting check

print("── Overfitting Check (train R² vs test R²) ───────────")
overfit = {}
ov_configs = [
    ("Linear Regression", lr, Xs_train, Xs_test),
    ("Random Forest",     rf, X_train,  X_test),
    ("Gradient Boosting", gb, X_train,  X_test),
]
for name, model, Xtr, Xte in ov_configs:
    tr_r2 = r2_score(y_train, model.predict(Xtr))
    te_r2 = r2_score(y_test,  model.predict(Xte))
    gap   = tr_r2 - te_r2
    flag  = "⚠ possible overfit" if gap > 0.02 else "✓"
    overfit[name] = (tr_r2, te_r2, gap)
    print(f"  {name:<22}  train={tr_r2:.4f}  test={te_r2:.4f}  "
          f"gap={gap:.4f}  {flag}")
print()


# 6. Edge case tests

print("── Edge Case Tests ───────────────────────────────────")
edge_cases = pd.DataFrame([
    {'sigma_s': 0.5, 'sigma_a': 0.3, 'sigma_f': 0.0,
     'sigma_t': 0.8, 'avg_collisions': 2.0, 'avg_scatters': 1.0,
     'expected': '~0.00', 'label': 'No fission (sigma_f=0)'},
    {'sigma_s': 0.3, 'sigma_a': 0.4, 'sigma_f': 0.4,
     'sigma_t': 1.1, 'avg_collisions': 1.5, 'avg_scatters': 0.5,
     'expected': '~0.50', 'label': 'High fission (sigma_f=0.4)'},
    {'sigma_s': 0.5, 'sigma_a': 0.3, 'sigma_f': 0.2,
     'sigma_t': 1.0, 'avg_collisions': 2.0, 'avg_scatters': 1.0,
     'expected': '~0.40', 'label': 'Typical mid-range'},
])
X_edge = edge_cases[FEATURES]
print(f"  {'Case':<30} {'Expected':>9}  {'LR':>7}  {'RF':>7}  {'GB':>7}")
print("  " + "─" * 65)
for i, row in edge_cases.iterrows():
    lp = lr.predict(scaler.transform(X_edge.iloc[[i]]))[0]
    rp = rf.predict(X_edge.iloc[[i]])[0]
    gp = gb.predict(X_edge.iloc[[i]])[0]
    print(f"  {row['label']:<30} {row['expected']:>9}  "
          f"{lp:>7.4f}  {rp:>7.4f}  {gp:>7.4f}")
print()


# 7. Save predictions CSV

pred_df = X_test.copy().reset_index(drop=True)
pred_df['actual_fission_rate'] = y_test.values
pred_df['lr_predicted']        = lr_res['pred']
pred_df['rf_predicted']        = rf_res['pred']
pred_df['gb_predicted']        = gb_res['pred']
pred_df['lr_residual']         = lr_res['resid']
pred_df['rf_residual']         = rf_res['resid']
pred_df['gb_residual']         = gb_res['resid']
pred_df['lr_abs_error']        = np.abs(lr_res['resid'])
pred_df['rf_abs_error']        = np.abs(rf_res['resid'])
pred_df['gb_abs_error']        = np.abs(gb_res['resid'])

pred_path = os.path.join(RESULTS_DIR, "predictions.csv")
pred_df.to_csv(pred_path, index=False)
print(f"Predictions saved → {pred_path}  "
      f"({os.path.getsize(pred_path)/1024:.1f} KB)")


# 8. Save metrics.txt

metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
with open(metrics_path, "w") as f:
    f.write("Monte Carlo Neutron Transport — ML Results\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Dataset rows  : {len(df):,}\n")
    f.write(f"Features      : {FEATURES}\n")
    f.write(f"Target        : {TARGET}\n")
    f.write(f"Train / Test  : {len(X_train):,} / {len(X_test):,}\n\n")
    f.write("── Test Set Results ──────────────────────────────\n")
    f.write(f"  {'Model':<22} {'R²':>7}  {'RMSE':>7}  {'MAE':>7}\n")
    f.write("  " + "─" * 42 + "\n")
    for r in results:
        f.write(f"  {r['name']:<22} {r['r2']:>7.4f}  "
                f"{r['rmse']:>7.4f}  {r['mae']:>7.4f}\n")
    f.write("\n── Cross-Validation (5-fold R²) ──────────────────\n")
    for name, scores in cv_scores.items():
        f.write(f"  {name:<22}  mean={scores.mean():.4f}  "
                f"std={scores.std():.4f}\n")
    f.write("\n── Overfitting Check ─────────────────────────────\n")
    for name, (tr, te, gap) in overfit.items():
        flag = "⚠" if gap > 0.02 else "✓"
        f.write(f"  {name:<22}  train={tr:.4f}  "
                f"test={te:.4f}  gap={gap:.4f}  {flag}\n")
    best = max(results, key=lambda x: x['r2'])
    f.write(f"\n── Best Model: {best['name']} ──\n")
    f.write(f"  R²={best['r2']:.4f}  RMSE={best['rmse']:.4f}  "
            f"MAE={best['mae']:.4f}\n")

print(f"Metrics saved  → {metrics_path}\n")


# 9. Plots

print("Building plots...")

colors = {
    'Linear Regression': '#e07b54',
    'Random Forest':     '#6abf69',
    'Gradient Boosting': '#5b8db8',
}

gb_imp = gb.feature_importances_
rf_imp = rf.feature_importances_
sorted_idx  = np.argsort(gb_imp)
feat_sorted = [FEATURES[i] for i in sorted_idx]

# ── Plot 1: ML dashboard
fig1 = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Predicted vs Actual — All Models",
        "Feature Importance: GB vs RF",
        "Residuals Distribution",
        "Model Comparison — R²",
    )
)

mn, mx = float(y_test.min()), float(y_test.max())

# (1,1) predicted vs actual
for r in results:
    fig1.add_trace(
        go.Scatter(
            x=y_test, y=r['pred'], mode='markers',
            marker=dict(color=colors[r['name']], size=3, opacity=0.35),
            name=r['name']
        ),
        row=1, col=1
    )
fig1.add_trace(
    go.Scatter(
        x=[mn, mx], y=[mn, mx], mode='lines',
        line=dict(color='white', dash='dash', width=1.5),
        name='Perfect fit', showlegend=False
    ),
    row=1, col=1
)

# (1,2) feature importance GB vs RF
fig1.add_trace(
    go.Bar(x=gb_imp[sorted_idx], y=feat_sorted, orientation='h',
           marker_color='#5b8db8', name='GB Importance', opacity=0.85),
    row=1, col=2
)
fig1.add_trace(
    go.Bar(x=rf_imp[sorted_idx], y=feat_sorted, orientation='h',
           marker_color='#6abf69', name='RF Importance', opacity=0.6),
    row=1, col=2
)

# (2,1) residuals
for r in results:
    fig1.add_trace(
        go.Histogram(
            x=r['resid'], nbinsx=50,
            marker_color=colors[r['name']], opacity=0.6,
            name=f"{r['name']} residuals"
        ),
        row=2, col=1
    )

# (2,2) R² comparison
for r in results:
    fig1.add_trace(
        go.Bar(
            x=[r['name']], y=[r['r2']],
            marker_color=colors[r['name']],
            text=[f"{r['r2']:.4f}"], textposition='outside',
            showlegend=False
        ),
        row=2, col=2
    )

fig1.update_layout(
    title_text="ML Pipeline — Monte Carlo Neutron Transport",
    title_font_size=16, height=850,
    template='plotly_dark', showlegend=True,
    barmode='overlay'
)
fig1.update_xaxes(title_text="Actual fission_rate",    row=1, col=1)
fig1.update_yaxes(title_text="Predicted fission_rate", row=1, col=1)
fig1.update_xaxes(title_text="Importance score",       row=1, col=2)
fig1.update_xaxes(title_text="Residual",               row=2, col=1)
fig1.update_yaxes(title_text="Count",                  row=2, col=1)
fig1.update_yaxes(title_text="R²",                     row=2, col=2)

out1 = os.path.join(RESULTS_DIR, "ml_dashboard.html")
fig1.write_html(out1)
print(f"Plot 1 saved → {out1}")

# ── Plot 2: 3D GB predictions
gb_resid = gb_res['resid']
fig2 = go.Figure(data=[
    go.Scatter3d(
        x=y_test.values, y=gb_res['pred'], z=gb_resid,
        mode='markers',
        marker=dict(
            size=3, color=gb_resid,
            colorscale='RdBu',
            colorbar=dict(title='Residual'),
            opacity=0.7
        ),
        text=[
            f"Actual: {a:.4f}<br>Predicted: {p:.4f}<br>Residual: {r:.4f}"
            for a, p, r in zip(y_test.values, gb_res['pred'], gb_resid)
        ],
        hoverinfo='text', name='GB Predictions'
    )
])
fig2.update_layout(
    title='3D — Actual vs Predicted vs Residual (Gradient Boosting)',
    scene=dict(
        xaxis_title='Actual fission_rate',
        yaxis_title='Predicted fission_rate',
        zaxis_title='Residual',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0))
    ),
    template='plotly_dark', height=700
)

out2 = os.path.join(RESULTS_DIR, "predictions_3d.html")
fig2.write_html(out2)
print(f"Plot 2 saved → {out2}")

import webbrowser
webbrowser.open(f"file://{os.path.abspath(out1)}")
webbrowser.open(f"file://{os.path.abspath(out2)}")

# ── Final summary
best = max(results, key=lambda x: x['r2'])
print("\n── Final Summary ─────────────────────────────────────")
print(f"  {'Model':<22} {'R²':>7}  {'RMSE':>7}  {'MAE':>7}")
print("  " + "─" * 45)
for r in results:
    tag = " ← best" if r['name'] == best['name'] else ""
    print(f"  {r['name']:<22} {r['r2']:>7.4f}  "
          f"{r['rmse']:>7.4f}  {r['mae']:>7.4f}{tag}")
print("─────────────────────────────────────────────────────")