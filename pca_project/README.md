# PCA Face Recognition — Step-by-Step Guide

## Folder Structure

```
pca_project/
├── download_dataset.py        ← Step 1: Download real ORL dataset
├── pca_face_recognition.py    ← Step 2: Run experiments, generate all figures
├── generate_latex_figures.py  ← Step 3: Generate LaTeX code for your report
├── requirements.txt           ← Python dependencies
├── README.md                  ← This file
└── results/                   ← Auto-created, all outputs go here
    ├── fig1_sample_faces.png
    ├── fig2_eigenfaces.png
    ├── fig3_reconstruction.png
    ├── fig4_accuracy_variance.png
    ├── fig5_scree.png
    ├── results_table.txt
    └── latex_section_X.tex
```

---

## Requirements

```
Python 3.8+
numpy
matplotlib
scikit-learn
Pillow
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Step-by-Step Instructions

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 2 — Download the real ORL dataset

**Requires internet connection.**

```bash
python download_dataset.py
```

This downloads the Olivetti Faces dataset (~1.4 MB) via scikit-learn
and saves two files:
- `faces_images.npy`  — shape (400, 64, 64), float32 in [0, 1]
- `faces_labels.npy`  — shape (400,), int in [0, 39]

> **If you already have the .npy files**, skip this step and place them
> in the `pca_project/` folder directly.

---

### Step 3 — Run the experiment

```bash
python pca_face_recognition.py
```

This script will:
1. Load and vectorise the images (64×64 → 4096-dim vectors)
2. Split into train (80%) / test (20%), stratified by subject
3. Build eigenfaces from scratch using the dual trick
4. Sweep k from 1 to 200 and evaluate 1-NN recognition accuracy
5. Save 5 figures + results table in `results/`

**Expected runtime: ~1–3 minutes** depending on your machine.

**Expected output (terminal):**
```
[Data]   Loaded 400 images, 40 subjects, image size 64×64
[Split]  Train=320  Test=80
[PCA]    Building up to 200 eigenfaces...
[Eval]   Sweeping k ...
  k=   1  →  Accuracy = xx.xx%
  k=   5  →  Accuracy = xx.xx%
  ...
  Best k = xx,  Best Accuracy = xx.xx%
All done!  Check the results/ folder.
```

---

### Step 4 — Generate LaTeX code

```bash
python generate_latex_figures.py
```

This creates `results/latex_section_X.tex` — a complete LaTeX section
ready to paste into your report's Section X.

---

### Step 5 — Insert results into your LaTeX report

1. Copy the `results/` folder into the same directory as your `.tex` file.

2. Open `results/latex_section_X.tex` and fill in the table values
   from `results/results_table.txt`.

3. Paste the content into your main `.tex` file at Section X.

4. Make sure these packages are in your preamble:
   ```latex
   \usepackage{graphicx}
   \usepackage{booktabs}
   \usepackage{float}
   \usepackage{tcolorbox}
   ```

---

## Output Files Explained

| File | Description |
|------|-------------|
| `fig1_sample_faces.png` | 10 sample subjects from the dataset |
| `fig2_eigenfaces.png` | Mean face + top 9 eigenfaces |
| `fig3_reconstruction.png` | Reconstruction quality at k=1,5,20,50,100 |
| `fig4_accuracy_variance.png` | Accuracy vs k + Cumulative variance vs k |
| `fig5_scree.png` | Scree plot (individual variance per PC) |
| `results_table.txt` | Full numerical results table |
| `latex_section_X.tex` | Ready-to-use LaTeX code for your report |

---

## Troubleshooting

**"faces_images.npy not found"**
→ Run `python download_dataset.py` first.

**Download fails (no internet)**
→ The script automatically uses a synthetic dataset for testing.
→ For the real dataset, download manually from:
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html

**Figures look blurry**
→ Normal for synthetic data. Use real ORL dataset for proper results.

**Slow evaluation**
→ Reduce `k_max` in `pca_face_recognition.py` line:
  `k_max = min(X_tr.shape[0] - 1, 100)`  ← change 200 to 100
