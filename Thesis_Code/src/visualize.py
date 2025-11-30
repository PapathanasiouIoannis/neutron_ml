# src/visualize.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from matplotlib.lines import Line2D
from src.physics import worker_get_plot_curve

def plot_physics_manifold(n_curves, baselines):
    print(f"\n--- Generating Physics Manifold Plots ---")
    seeds = np.random.randint(0, 1e9, n_curves)
    
    res_h = Parallel(n_jobs=-1)(delayed(worker_get_plot_curve)('hadronic', baselines, s) for s in tqdm(seeds, desc="H-EoS"))
    res_q = Parallel(n_jobs=-1)(delayed(worker_get_plot_curve)('quark', baselines, s) for s in tqdm(seeds, desc="Q-EoS"))
    
    curves_h = [c for c in res_h if c is not None]
    curves_q = [c for c in res_q if c is not None]

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    STYLE_H = {'color': 'blue', 'alpha': 0.02, 'lw': 1}
    STYLE_Q = {'color': 'red',  'alpha': 0.02, 'lw': 1}

    for c in curves_h:
        log_L = np.log10(c[:,2])
        ax[0].plot(c[:,1], c[:,0], **STYLE_H)
        ax[1].plot(c[:,0], log_L, **STYLE_H)
        ax[2].plot(c[:,1], log_L, **STYLE_H)

    for c in curves_q:
        log_L = np.log10(c[:,2])
        ax[0].plot(c[:,1], c[:,0], **STYLE_Q)
        ax[1].plot(c[:,0], log_L, **STYLE_Q)
        ax[2].plot(c[:,1], log_L, **STYLE_Q)

    ax[0].set_xlabel("Radius (km)"); ax[0].set_ylabel(r"Mass ($M_{\odot}$)")
    ax[0].set_xlim(8, 16); ax[0].set_ylim(0.5, 3.0)
    ax[0].axhline(1.97, color='k', linestyle=':', label="J0740")
    ax[0].axvspan(0, 11.0, color='gray', alpha=0.1)
    
    ax[1].set_xlabel(r"Mass ($M_{\odot}$)"); ax[1].set_ylabel(r"Log($\Lambda$)")
    ax[1].set_xlim(1.0, 2.6); ax[1].set_ylim(0, 4)
    ax[1].axhline(np.log10(580), color='k', linestyle='--')
    
    ax[2].set_xlabel("Radius (km)"); ax[2].set_ylabel(r"Log($\Lambda$)")
    ax[2].set_xlim(8, 16); ax[2].set_ylim(0, 4)

    custom_lines = [Line2D([0], [0], color='blue', lw=2), Line2D([0], [0], color='red', lw=2)]
    ax[0].legend(custom_lines, ['Hadronic', 'Quark'], loc='upper left')
    
    fig.tight_layout()
    plt.savefig("plots/fig_physics_manifold.png", dpi=300)
    plt.close()
    print("Saved plots/fig_physics_manifold.png")

def plot_diagnostics(model, X_test, y_test):
    print("\n--- Generating ML Diagnostic Plots ---")
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    # 1. Confusion Matrix & ROC
    fig, ax = plt.subplots(1, 3, figsize=(20, 6))
    
    # CM
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Hadronic', 'Quark']).plot(cmap='Blues', ax=ax[0], colorbar=False)
    ax[0].set_title("Confusion Matrix")

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
    ax[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
    ax[1].set_xlabel('False Positive Rate'); ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('ROC Curve'); ax[1].legend()

    # Confidence Hist
    df_probs = pd.DataFrame({'Probability': y_probs, 'Label': y_test})
    df_probs['Label'] = df_probs['Label'].map({0: 'Hadronic', 1: 'Quark'})
    sns.histplot(data=df_probs, x='Probability', hue='Label', element="step", stat="density", common_norm=False, bins=30, palette={'Hadronic': 'blue', 'Quark': 'red'}, ax=ax[2])
    ax[2].set_title('Model Confidence')
    
    plt.tight_layout()
    plt.savefig("plots/fig_ml_diagnostics.png", dpi=300)
    plt.close()

    # 2. Violin Plots
    plot_df = X_test.copy()
    plot_df['Label'] = y_test.map({0: 'Hadronic', 1: 'Quark'})
    bins = [0.0, 1.1, 1.7, 3.0]
    labels = ['Low (<1.1)', 'Canonical (1.1-1.7)', 'High (>1.7)']
    plot_df['Mass_Bin'] = pd.cut(plot_df['Mass'], bins=bins, labels=labels)
    
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    sns.violinplot(data=plot_df, x='Mass_Bin', y='Radius', hue='Label', split=True, inner='quart', palette={'Hadronic': 'blue', 'Quark': 'red'}, ax=ax[0])
    ax[0].set_title('Radius Separation by Mass')
    sns.violinplot(data=plot_df, x='Mass_Bin', y='LogLambda', hue='Label', split=True, inner='quart', palette={'Hadronic': 'blue', 'Quark': 'red'}, ax=ax[1])
    ax[1].set_title('Tidal Deformability Separation by Mass')
    
    plt.tight_layout()
    plt.savefig("plots/fig_violin_physics.png", dpi=300)
    plt.close()
    print("Saved plots/fig_ml_diagnostics.png and plots/fig_violin_physics.png")