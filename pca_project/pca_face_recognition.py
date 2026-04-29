"""
PCA-Based Face Recognition — From Scratch
==========================================
Dataset : ORL (Olivetti) Face Database  — 40 subjects, 10 images each
          Load via:  python download_dataset.py   (requires internet)
          OR place faces_images.npy + faces_labels.npy in this folder.

Usage:
    python pca_face_recognition.py

Outputs (saved to  results/):
    fig1_sample_faces.png
    fig2_mean_face.png
    fig3_eigenfaces.png
    fig4_reconstruction.png
    fig5_accuracy_vs_k.png
    fig6_variance_explained.png
    results_table.txt
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
#  0.  Setup
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs('results', exist_ok=True)
SEED = 42
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
#  1.  Load Data
# ─────────────────────────────────────────────────────────────────────────────
def load_data():
    """
    Load face images.
    Expected: faces_images.npy  shape (N, H, W)  float in [0,1]
              faces_labels.npy  shape (N,)        int
    """
    if not os.path.exists('faces_images.npy'):
        raise FileNotFoundError(
            "faces_images.npy not found.\n"
            "Run:  python download_dataset.py\n"
            "Or place the .npy files in this folder."
        )
    X = np.load('faces_images.npy')   # (N, H, W)
    y = np.load('faces_labels.npy')   # (N,)
    print(f"[Data] Loaded {X.shape[0]} images, "
          f"{len(np.unique(y))} subjects, "
          f"image size {X.shape[1]}×{X.shape[2]}")
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
#  2.  Preprocessing — Vectorise + Flatten
# ─────────────────────────────────────────────────────────────────────────────
def vectorise(X):
    """
    Flatten each image H×W → vector of length d = H*W.
    Input  : X  (N, H, W)
    Output : Xv (N, d)
    """
    N = X.shape[0]
    return X.reshape(N, -1).astype(np.float64)   # (N, d)


# ─────────────────────────────────────────────────────────────────────────────
#  3.  PCA — Eigenface Construction (from scratch, no sklearn PCA)
# ─────────────────────────────────────────────────────────────────────────────
def build_eigenfaces(X_train, k_max):
    """
    Compute up to k_max eigenfaces from training data.

    Parameters
    ----------
    X_train : ndarray (N_tr, d)
    k_max   : int  — number of eigenfaces to compute

    Returns
    -------
    U_k      : ndarray (d, k_max)  — eigenface matrix (columns = eigenfaces)
    mu       : ndarray (d,)        — mean face
    eigvals  : ndarray (k_max,)    — eigenvalues (descending)
    """
    N, d = X_train.shape

    # Step 1 — Mean face
    mu = X_train.mean(axis=0)                      # (d,)

    # Step 2 — Centre the data
    X_c = X_train - mu                             # (N, d)

    # Step 3 — Gram matrix  L = X̃ X̃ᵀ / (N-1)  ∈ ℝ^{N×N}
    #          (Dual / compact trick: N << d)
    L = X_c @ X_c.T / (N - 1)                     # (N, N)

    # Step 4 — Eigendecomposition of L
    eigvals_L, V = np.linalg.eigh(L)              # ascending order

    # Step 5 — Sort descending
    idx = np.argsort(eigvals_L)[::-1]
    eigvals_L = eigvals_L[idx]
    V = V[:, idx]                                  # (N, N)

    k_use = min(k_max, N - 1)

    # Step 6 — Map eigenvectors back to ℝ^d space
    U = X_c.T @ V[:, :k_use]                      # (d, k_use)

    # Step 7 — Normalise each column
    norms = np.linalg.norm(U, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    U = U / norms                                  # (d, k_use)

    return U, mu, eigvals_L[:k_use]


# ─────────────────────────────────────────────────────────────────────────────
#  4.  Projection
# ─────────────────────────────────────────────────────────────────────────────
def project(X, U_k, mu):
    """
    Project images onto PCA subspace.
    X    : (N, d)
    U_k  : (d, k)
    mu   : (d,)
    → Z  : (N, k)
    """
    return (X - mu) @ U_k      # (N, k)


def reconstruct(Z, U_k, mu):
    """
    Reconstruct images from PCA codes.
    Z    : (N, k)
    U_k  : (d, k)
    mu   : (d,)
    → X_hat : (N, d)
    """
    return Z @ U_k.T + mu      # (N, d)


# ─────────────────────────────────────────────────────────────────────────────
#  5.  Recognition — 1-Nearest Neighbour (Euclidean)
# ─────────────────────────────────────────────────────────────────────────────
def recognize_1nn(Z_test, Z_train, y_train):
    """
    1-NN classifier in PCA space.
    Z_test  : (N_te, k)
    Z_train : (N_tr, k)
    y_train : (N_tr,)
    → preds : (N_te,)
    """
    preds = []
    for z in Z_test:
        dists = np.linalg.norm(Z_train - z, axis=1)   # (N_tr,)
        preds.append(y_train[np.argmin(dists)])
    return np.array(preds)


def evaluate_k(X_train, y_train, X_test, y_test, U_all, mu, k):
    """Accuracy for a given k."""
    U_k    = U_all[:, :k]
    Z_tr   = project(X_train, U_k, mu)
    Z_te   = project(X_test,  U_k, mu)
    preds  = recognize_1nn(Z_te, Z_tr, y_train)
    return np.mean(preds == y_test) * 100.0


# ─────────────────────────────────────────────────────────────────────────────
#  6.  Figures
# ─────────────────────────────────────────────────────────────────────────────
IMG_H = IMG_W = 64   # will be set from data

def show_grid(images, titles, rows, cols, fname, main_title='',
              cmap='gray', figsize=None):
    figsize = figsize or (cols * 2.2, rows * 2.5)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(rows, cols)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i], cmap=cmap, vmin=0, vmax=1)
            if titles:
                ax.set_title(titles[i], fontsize=8)
        ax.axis('off')
    if main_title:
        fig.suptitle(main_title, fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved  {fname}")


def fig_sample_faces(X_img, y, fname='results/fig1_sample_faces.png'):
    """Show first image of each of 10 random subjects."""
    subjects = np.unique(y)[:10]
    imgs, titles = [], []
    for s in subjects:
        idx = np.where(y == s)[0][0]
        imgs.append(X_img[idx])
        titles.append(f'Subject {s+1}')
    show_grid(imgs, titles, 2, 5, fname,
              main_title='Sample Faces — ORL Dataset')


def fig_mean_and_eigenfaces(mu, U_k, H, W,
                             fname='results/fig2_eigenfaces.png'):
    """Mean face + top-9 eigenfaces."""
    imgs   = [mu.reshape(H, W)]
    titles = ['Mean Face']
    for j in range(min(9, U_k.shape[1])):
        ef = U_k[:, j].reshape(H, W)
        ef = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
        imgs.append(ef)
        titles.append(f'Eigenface {j+1}')
    show_grid(imgs, titles, 2, 5, fname,
              main_title='Mean Face and Top-9 Eigenfaces')


def fig_reconstruction(X_img, y, U_all, mu, H, W,
                        fname='results/fig3_reconstruction.png'):
    """Reconstruction quality at k = 1, 5, 20, 50, 100, original."""
    k_vals  = [1, 5, 20, 50, 100]
    idx     = np.where(y == 0)[0][0]        # first image of subject 0
    x_orig  = X_img[idx].reshape(1, -1).astype(np.float64)

    n_cols  = len(k_vals) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 2.2, 2.8))

    axes[0].imshow(X_img[idx], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original', fontsize=9, fontweight='bold')
    axes[0].axis('off')

    for i, k in enumerate(k_vals):
        U_k  = U_all[:, :k]
        Z    = project(x_orig, U_k, mu)
        xhat = reconstruct(Z, U_k, mu).reshape(H, W)
        xhat = np.clip(xhat, 0, 1)
        mse  = np.mean((X_img[idx] - xhat) ** 2)
        axes[i+1].imshow(xhat, cmap='gray', vmin=0, vmax=1)
        axes[i+1].set_title(f'k={k}\nMSE={mse:.4f}', fontsize=8)
        axes[i+1].axis('off')

    fig.suptitle('Face Reconstruction Quality vs. k',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved  {fname}")


def fig_accuracy_and_variance(k_values, accuracies, eigvals,
                               fname='results/fig4_accuracy_variance.png'):
    """Side-by-side: accuracy vs k  AND  cumulative variance vs k."""
    total_var = eigvals.sum()
    cumvar    = np.cumsum(eigvals) / total_var * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # — Accuracy curve
    ax1.plot(k_values, accuracies, 'o-', color='steelblue',
             linewidth=2, markersize=5, label='Test Accuracy')
    ax1.axhline(max(accuracies), color='red', linestyle='--', linewidth=1,
                label=f'Peak = {max(accuracies):.1f}%  (k={k_values[np.argmax(accuracies)]})')
    ax1.set_xlabel('Number of Principal Components  $k$', fontsize=11)
    ax1.set_ylabel('Recognition Accuracy (%)', fontsize=11)
    ax1.set_title('Face Recognition Accuracy vs. $k$', fontsize=12,
                  fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 105)

    # — Variance curve
    ks = np.arange(1, len(eigvals) + 1)
    ax2.plot(ks, cumvar, color='tomato', linewidth=2)
    ax2.axhline(95, color='gray',  linestyle='--', linewidth=1,
                label='95% threshold')
    ax2.axhline(99, color='black', linestyle=':',  linewidth=1,
                label='99% threshold')
    k95 = int(np.searchsorted(cumvar, 95)) + 1
    k99 = int(np.searchsorted(cumvar, 99)) + 1
    ax2.axvline(k95, color='gray',  linestyle='--', linewidth=1,
                label=f'k={k95} → 95%')
    ax2.axvline(k99, color='black', linestyle=':',  linewidth=1,
                label=f'k={k99} → 99%')
    ax2.set_xlabel('Number of Principal Components  $k$', fontsize=11)
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=11)
    ax2.set_title('Cumulative Explained Variance vs. $k$', fontsize=12,
                  fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 102)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved  {fname}")


def fig_scree_plot(eigvals, n=50, fname='results/fig5_scree.png'):
    """Scree plot — first n eigenvalues."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, n+1), eigvals[:n] / eigvals.sum() * 100,
           color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_xlabel('Principal Component Index', fontsize=11)
    ax.set_ylabel('Variance Explained (%)', fontsize=11)
    ax.set_title(f'Scree Plot — First {n} Eigenvalues', fontsize=12,
                 fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4, axis='y')
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved  {fname}")


# ─────────────────────────────────────────────────────────────────────────────
#  7.  Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PCA Face Recognition — from scratch")
    print("=" * 60)

    # 7.1  Load & vectorise
    X_img, y = load_data()
    H, W     = X_img.shape[1], X_img.shape[2]
    Xv       = vectorise(X_img)   # (400, 4096)

    # 7.2  Train / test split (stratified, 80/20)
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xv, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"[Split]  Train={X_tr.shape[0]}  Test={X_te.shape[0]}")

    # 7.3  Figure 1 — sample faces
    print("\n[Figures] Generating...")
    fig_sample_faces(X_img, y)

    # 7.4  Build eigenfaces (compute up to N_tr-1 components)
    k_max = min(X_tr.shape[0] - 1, 200)
    print(f"[PCA]  Building up to {k_max} eigenfaces...")
    U_all, mu, eigvals = build_eigenfaces(X_tr, k_max)
    print(f"[PCA]  Eigenface matrix shape: {U_all.shape}")

    # 7.5  Figure 2 — mean face + top eigenfaces
    # Normalise mu to [0,1] for display
    mu_disp = (mu - mu.min()) / (mu.max() - mu.min() + 1e-8)
    fig_mean_and_eigenfaces(mu_disp, U_all, H, W)

    # 7.6  Figure 3 — reconstruction quality
    fig_reconstruction(X_img, y, U_all, mu, H, W)

    # 7.7  Sweep k — accuracy evaluation
    k_values = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                + list(range(15, 51, 5))
                + list(range(60, k_max + 1, 20)))
    k_values = [k for k in k_values if k <= k_max]

    print(f"\n[Eval]  Sweeping k over {k_values} ...")
    accuracies = []
    for k in k_values:
        acc = evaluate_k(X_tr, y_tr, X_te, y_te, U_all, mu, k)
        accuracies.append(acc)
        print(f"  k={k:4d}  →  Accuracy = {acc:.2f}%")

    best_k   = k_values[int(np.argmax(accuracies))]
    best_acc = max(accuracies)
    print(f"\n  Best k = {best_k},  Best Accuracy = {best_acc:.2f}%")

    # 7.8  Figure 4 — accuracy + variance
    fig_accuracy_and_variance(k_values, accuracies, eigvals)

    # 7.9  Figure 5 — scree plot
    fig_scree_plot(eigvals, n=min(50, len(eigvals)))

    # 7.10  Save results table
    total_var = eigvals.sum()
    cumvar    = np.cumsum(eigvals) / total_var * 100
    k95 = int(np.searchsorted(cumvar, 95)) + 1
    k99 = int(np.searchsorted(cumvar, 99)) + 1

    with open('results/results_table.txt', 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("  PCA FACE RECOGNITION — RESULTS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset     : ORL (Olivetti) — 40 subjects × 10 images\n")
        f.write(f"Image size  : {H}×{W} = {H*W} dimensions\n")
        f.write(f"Train/Test  : {X_tr.shape[0]} / {X_te.shape[0]} images\n")
        f.write(f"Max k used  : {k_max}\n\n")
        f.write(f"k for 95% variance : {k95}\n")
        f.write(f"k for 99% variance : {k99}\n\n")
        f.write(f"Best k             : {best_k}\n")
        f.write(f"Best Accuracy      : {best_acc:.2f}%\n\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'k':>6}  {'Accuracy (%)':>14}  {'Cum. Var (%)':>14}\n")
        f.write("-" * 40 + "\n")
        for k, acc in zip(k_values, accuracies):
            cv = cumvar[k-1] if k <= len(cumvar) else cumvar[-1]
            f.write(f"{k:>6}  {acc:>14.2f}  {cv:>14.2f}\n")
        f.write("-" * 40 + "\n")

    print("\n  Saved  results/results_table.txt")
    print("\n" + "=" * 60)
    print("  All done!  Check the  results/  folder.")
    print("=" * 60)

    return k_values, accuracies, eigvals


if __name__ == '__main__':
    main()
