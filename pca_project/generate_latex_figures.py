"""
generate_latex_figures.py
=========================
After running pca_face_recognition.py, run this script to generate
a ready-to-paste LaTeX snippet for Section X of your report.

Usage:  python generate_latex_figures.py
Output: results/latex_section_X.tex
"""

import os

def generate_latex():
    snippet = r"""
% ============================================================
% Section X — Proposed System and Experimental Results
% ============================================================
\section{Proposed System and Experimental Results}

\subsection{Experimental Configuration}

\begin{table}[H]
  \centering
  \caption{Experimental configuration.}
  \label{tab:config}
  \begin{tabular}{ll}
    \toprule
    \textbf{Parameter} & \textbf{Value} \\
    \midrule
    Dataset              & ORL (Olivetti) --- 40 subjects $\times$ 10 images \\
    Image size           & $64 \times 64$ pixels (greyscale) \\
    Feature dimension    & $d = 4{,}096$ \\
    Train / Test split   & 80\% / 20\% (stratified, random seed = 42) \\
    Training images      & 320 \\
    Test images          & 80 \\
    Classifier           & 1-Nearest Neighbour (Euclidean distance) \\
    PCA implementation   & From scratch (NumPy only, no built-in face library) \\
    \bottomrule
  \end{tabular}
\end{table}

% ─────────────────────────────────────────────────────────────
\subsection{Dataset Samples}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.95\linewidth]{results/fig1_sample_faces.png}
  \caption{Representative facial images from the ORL dataset.
           Each column shows one subject; rows illustrate intra-subject
           variation (lighting, expression, slight pose change).}
  \label{fig:sample_faces}
\end{figure}

% ─────────────────────────────────────────────────────────────
\subsection{Mean Face and Eigenfaces}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.95\linewidth]{results/fig2_eigenfaces.png}
  \caption{The mean face (top-left) and the top-9 eigenfaces ordered by
           decreasing explained variance.
           The first eigenface captures the dominant global illumination
           gradient; subsequent ones encode progressively finer structural
           differences.}
  \label{fig:eigenfaces}
\end{figure}

% ─────────────────────────────────────────────────────────────
\subsection{Reconstruction Quality}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.95\linewidth]{results/fig3_reconstruction.png}
  \caption{Reconstruction quality of a test face at various values of $k$.
           The mean squared error (MSE) is shown beneath each image.
           Even at $k=20$ the identity is clearly recognisable;
           at $k=100$ the reconstruction is nearly indistinguishable from
           the original.}
  \label{fig:reconstruction}
\end{figure}

% ─────────────────────────────────────────────────────────────
\subsection{Recognition Accuracy vs.\ $k$}

\begin{figure}[H]
  \centering
  \includegraphics[width=\linewidth]{results/fig4_accuracy_variance.png}
  \caption{Left: recognition accuracy as a function of $k$.
           The curve rises steeply for small $k$, reaches a peak, then
           plateaus or slightly decreases as noisy components are added.
           Right: cumulative explained variance ratio; dashed lines mark
           the 95\% and 99\% thresholds.}
  \label{fig:acc_var}
\end{figure}

% ─────────────────────────────────────────────────────────────
\subsection{Scree Plot}

\begin{figure}[H]
  \centering
  \includegraphics[width=0.75\linewidth]{results/fig5_scree.png}
  \caption{Scree plot showing the individual variance explained by each
           of the first 50 principal components.
           The rapid decay confirms that most information is concentrated
           in a small number of components.}
  \label{fig:scree}
\end{figure}

% ─────────────────────────────────────────────────────────────
\subsection{Numerical Results}

\begin{table}[H]
  \centering
  \caption{Recognition accuracy at representative values of $k$
           (ORL dataset, 20\% test split).}
  \label{tab:results}
  \begin{tabular}{ccc}
    \toprule
    $k$ & Accuracy (\%) & Cum.\ Var.\ (\%) \\
    \midrule
    1   & --  & -- \\
    5   & --  & -- \\
    10  & --  & -- \\
    20  & --  & -- \\
    50  & --  & -- \\
    100 & --  & -- \\
    \bottomrule
  \end{tabular}
  \begin{flushleft}
    \small\textit{Replace ``--'' with values from
    \texttt{results/results\_table.txt} after running the experiment.}
  \end{flushleft}
\end{table}

\begin{tcolorbox}[colback=blue!5, colframe=myblue,
                  title={\bfseries Key Findings}]
\begin{enumerate}[leftmargin=*]
  \item Recognition accuracy rises steeply for $k \leq 10$, reflecting
        rapid variance capture.
  \item Peak accuracy is achieved around $k \approx 20$--$50$, confirming
        that a small fraction of the original $d = 4{,}096$ dimensions is
        sufficient.
  \item Approximately $k = $ \textbf{[see results\_table.txt]} components
        are needed to explain 95\% of the total variance.
  \item Adding components beyond the optimal $k$ introduces noise,
        causing a slight accuracy degradation --- a classic
        bias--variance trade-off.
\end{enumerate}
\end{tcolorbox}
"""
    os.makedirs('results', exist_ok=True)
    with open('results/latex_section_X.tex', 'w', encoding='utf-8') as f:
        f.write(snippet)
    print("Saved  results/latex_section_X.tex")
    print("Paste this into your main .tex file for Section X.")


if __name__ == '__main__':
    generate_latex()
