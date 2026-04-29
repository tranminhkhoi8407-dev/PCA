"""
download_dataset.py
===================
Downloads the real ORL (Olivetti) Face Dataset via scikit-learn.

Requirements:  pip install scikit-learn numpy
Run          :  python download_dataset.py

Output       :  faces_images.npy   (400, 64, 64)  float32
                faces_labels.npy   (400,)          int
"""

import numpy as np

def download_olivetti():
    try:
        from sklearn.datasets import fetch_olivetti_faces
    except ImportError:
        print("ERROR: scikit-learn not installed.")
        print("  Run:  pip install scikit-learn")
        return

    print("Downloading Olivetti Faces dataset...")
    print("(~1.4 MB — may take a few seconds)")

    data = fetch_olivetti_faces(shuffle=False, random_state=42)

    X = data.images   # (400, 64, 64) float32 in [0, 1]
    y = data.target   # (400,) int in [0, 39]

    np.save('faces_images.npy', X)
    np.save('faces_labels.npy', y)

    print(f"\nSaved successfully!")
    print(f"  faces_images.npy  shape: {X.shape}")
    print(f"  faces_labels.npy  shape: {y.shape}")
    print(f"  Subjects: {len(np.unique(y))}")
    print(f"  Images per subject: {np.bincount(y)[0]}")
    print(f"  Image size: {X.shape[1]}x{X.shape[2]}")
    print("\nNow run:  python pca_face_recognition.py")


if __name__ == '__main__':
    download_olivetti()
