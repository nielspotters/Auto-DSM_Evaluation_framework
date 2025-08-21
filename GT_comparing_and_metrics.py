"""
GT_comparing_and_metrics
------------------------
This script compares generated DSMs (Design Structure Matrices) against a Ground Truth (GT) DSM,
aligns their component labels, computes evaluation metrics, and exports structured results.

Main functionality:
1. Load Ground Truth DSM and generated DSMs.
2. Align component labels using fuzzy matching + optional manual correction.
3. Compute evaluation metrics (accuracy, completeness, correctness, etc.).
4. Save results to structured Excel sheets + optional heatmap visualizations.

Expected folder structure (relative to this script):
- GT_DSM/              → contains one GT_DSM_*.xlsx file
- Data_xlsx/           → contains generated DSM files (Koh_DSM_*.xlsx)
- DSM_Evaluation_Results/ (created automatically for outputs)

Usage:
    python GT_comparing_and_metrics.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rapidfuzz import process, fuzz
from matplotlib.colors import ListedColormap

# ---------------- Environment Check (Optional) ----------------
# If you want to enforce a specific virtual environment, set the path below.
# For public sharing, this is just a warning instead of a hard stop.
EXPECTED_VENV_PATH = None  # Example: r"C:\path\to\venv\Scripts\python.exe"

if EXPECTED_VENV_PATH:
    if os.path.normcase(sys.executable) != os.path.normcase(EXPECTED_VENV_PATH):
        print(
            f"\n[Warning] Script not running in expected virtual environment.\n"
            f"Expected: {EXPECTED_VENV_PATH}\n"
            f"Got     : {sys.executable}\n"
            f"Proceeding anyway...\n"
        )

# Pandas config (suppress warnings for future downcasting changes)
pd.set_option('future.no_silent_downcasting', False)

# ---------------- Define Base Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GT_DSM_DIR = os.path.join(BASE_DIR, 'GT_DSM')
GEN_DSM_DIR = os.path.join(BASE_DIR, 'Data_xlsx')
OUTPUT_DIR = os.path.join(BASE_DIR, 'DSM_Evaluation_Results')

# Locate GT DSM file (expecting exactly one)
gt_dsm_files = glob.glob(os.path.join(GT_DSM_DIR, 'GT_DSM_*.xlsx'))
if not gt_dsm_files:
    raise FileNotFoundError("No GT_DSM_*.xlsx file found in GT_DSM/ folder.")
GT_DSM_PATH = gt_dsm_files[0]

print(f"Using GT DSM: {os.path.basename(GT_DSM_PATH)}")

# Recognized naming patterns for generated DSMs
GEN_PATTERNS = ['GEN_DSM', 'Koh_DSM']


# ---------------- Alignment and Cleaning ----------------
def interactive_align_and_clean_dsms(gt_df, gen_df):
    """
    Align and clean DSMs by matching component labels between GT and GEN.

    Uses fuzzy string matching to propose alignments, and allows user confirmation
    or rejection interactively. This ensures robustness when component labels differ.

    Returns:
        tuple: (gt_elements, gen_elements, matched_elements, gt_clean, gen_clean)
    """
    # Strip whitespace and normalize labels
    gt_df.index = gt_df.index.map(lambda s: str(s).strip())
    gt_df.columns = gt_df.columns.map(lambda s: str(s).strip())
    gen_df.index = gen_df.index.map(lambda s: str(s).strip())
    gen_df.columns = gen_df.columns.map(lambda s: str(s).strip())

    gt_elements = sorted(list(gt_df.index), key=lambda x: x.lower())
    gen_elements = sorted(list(gen_df.index), key=lambda x: x.lower())

    matched_map = {}          # GT label -> GEN label
    used_gen_elements = set()

    print("\n--- Starting interactive fuzzy alignment of components ---\n")

    for gt in gt_elements:
        normalized_gt = gt.strip().lower()
        candidates = [g for g in gen_elements if g not in used_gen_elements]

        match, score, _ = process.extractOne(gt, candidates, scorer=fuzz.token_sort_ratio)

        # If perfect or near-perfect match, auto-accept
        if normalized_gt == match.strip().lower():
            matched_map[gt] = match
            used_gen_elements.add(match)
            print(f"Auto-matched: '{gt}' ↔ '{match}' [Score: {score}]")
            continue

        # Otherwise, ask user for confirmation
        print(f"\nGT: '{gt}'   ↔   GEN: '{match}'   [Match Score: {score}]")
        decision = input("→ Do these refer to the same component? [y/n] \n> ").strip().lower()

        if decision in ('y', ''):
            matched_map[gt] = match
            used_gen_elements.add(match)
            print(f"→ Recorded match: {gt} ↔ {match}")
        elif decision == 'n':
            print_element_table(gt_elements.copy(), gen_elements.copy())
            while True:
                choice = input("→ Which one should be removed? [GT/GEN] \n> ").strip().lower()
                if choice == 'gt':
                    print(f"→ '{gt}' will be excluded from analysis.")
                    break
                elif choice == 'gen':
                    used_gen_elements.add(match)
                    print(f"→ '{match}' will be excluded from analysis.")
                    break
                else:
                    print("Invalid input. Please enter 'GT' or 'GEN'.")
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

    final_gt_elements = list(matched_map.keys())
    original_gen_elements = list(matched_map.values())

    if not final_gt_elements:
        print("\n[Critical] No matching components found. Cannot proceed.")
        return [], pd.DataFrame(), pd.DataFrame()

    # Extract cleaned/realigned matrices
    gt_clean = gt_df.loc[final_gt_elements, final_gt_elements].copy()
    gen_clean = gen_df.loc[original_gen_elements, original_gen_elements].copy()

    # Rename GEN elements to GT naming convention
    rename_map = dict(zip(original_gen_elements, final_gt_elements))
    gen_clean.rename(index=rename_map, columns=rename_map, inplace=True)
    gen_clean = gen_clean.reindex(index=final_gt_elements, columns=final_gt_elements)

    print(f"\nOriginal GT components: {len(gt_elements)}")
    print(f"Original GEN components: {len(gen_elements)}")
    print(f"Final aligned components: {len(final_gt_elements)}")

    return gt_elements, gen_elements, final_gt_elements, gt_clean, gen_clean


def print_element_table(gt_elements, gen_elements):
    """
    Helper function to print side-by-side comparison of GT and GEN element lists.
    """
    max_len = max(len(gt_elements), len(gen_elements))
    gt_elements += [''] * (max_len - len(gt_elements))
    gen_elements += [''] * (max_len - len(gen_elements))

    print("\nCurrent component lists:\n")
    print(f"{'GT Elements':<30} {'GEN Elements':<30}")
    print('-' * 60)
    for gt, gen in zip(gt_elements, gen_elements):
        print(f"{gt:<30} {gen:<30}")
    print()


# ---------------- Evaluation Metrics ----------------
def calculate_evaluation_metrics(gt, gen, gt_elements, gen_elements):
    """
    Compute DSM evaluation metrics between GT and GEN matrices.

    Metrics include:
    - True/False Positives/Negatives
    - Selective Accuracy
    - Completeness, Correctness
    - NZF
    - Average Penalty (FP, FN, IDK weighted)
    """
    n_elements = gen.shape[0]
    gen_matrix = gen
    gt_matrix = gt

    # Confusion matrix components (excluding diagonal)
    TP = np.sum((gt_matrix == 1) & (gen_matrix == 1)) - n_elements
    TN = np.sum((gt_matrix == -1) & (gen_matrix == -1))
    FP = np.sum((gt_matrix == -1) & (gen_matrix == 1))
    FN = np.sum((gt_matrix == 1) & (gen_matrix == -1))
    IDK = np.sum(gen_matrix == 0)
    total = gt_matrix.size - n_elements

    # DSM metrics
    count_error = np.sum(gen_matrix == 9)
    count_symmetric = 2 * sum(
        gen_matrix[i, j] == 1 and gen_matrix[j, i] == 1
        for i in range(n_elements)
        for j in range(i + 1, n_elements))
    count_active = TP + FP
    count_inactive = TN + FN

    completeness = ((count_active + count_inactive) /
                    (count_active + count_inactive + IDK + count_error) * 100
                    if (count_active + count_inactive + IDK + count_error) else 0)

    correctness = count_symmetric / count_active * 100 if count_active else 0
    NZF = count_active / (n_elements * (n_elements - 1)) * 100

    # Classification metrics
    SA = (TP + TN) / (total - IDK) if (total - IDK) > 0 else np.nan
    Accuracy = (TP + TN) / total if total > 0 else np.nan

    # Penalty model (weighted penalties for FP, FN, IDK)
    penalty_FP = 1.0
    penalty_FN = 1.0
    penalty_IDK = 0.5
    avg_penalty = (penalty_FP * FP + penalty_FN * FN + penalty_IDK * IDK) / total

    return {
        "Total GEN Elements": len(gen_elements),
        "Total GT Elements": len(gt_elements),
        "Matched Elements": int(n_elements),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN, "IDK": IDK,
        "Error entries": int(count_error),
        "Symmetric entries": int(count_symmetric),
        "Selective Accuracy": round(SA, 4),
        "Accuracy": round(Accuracy, 4),
        "Average Penalty": round(avg_penalty, 4),
        "Correctness [% of present entries]": round(correctness, 2),
        "Completeness [% of total entries]": round(completeness, 2),
        "NZF [% of total entries]": round(NZF, 2),
    }

# ---------------- Visualization ----------------
def generate_dsm_heatmaps(matched_matrix, mismatch_matrix, uncertainty_matrix, element_labels, output_prefix):
    """
    Generating matched, mismatched and uncertainty heatmaps
    """
    def _format_y_labels(labels):
        return [f"{i+1}: {label}" for i, label in enumerate(labels)]  # Start index from 1

    def _plot_heatmap(data, title, filename, cmap, vmin=None, vmax=None, special_zero_gray=False, colorbar_label="Value"):
        plt.figure(figsize=(10, 8))
        heatmap_data = data.replace("", np.nan).infer_objects(copy=False).astype(float)

        if special_zero_gray:
            base_cmap = plt.get_cmap(cmap)
            colors = base_cmap(np.linspace(0, 1, 256))
            colors[128] = [0.7, 0.7, 0.7, 1.0]  # gray for 0
            cmap = ListedColormap(colors)

        sns.heatmap(
            heatmap_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".0f",
            cbar_kws={"label": colorbar_label},
            linewidths=0.5,
            linecolor="gray",
            square=True,
            xticklabels=[str(i + 1) for i in range(len(element_labels))],
            yticklabels=_format_y_labels(element_labels)
        )

        plt.title(title)
        plt.xlabel("Component Index (j)")
        plt.ylabel("Component Index and Name (i)")
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    # Matched Heatmap
    _plot_heatmap(
        matched_matrix,
        title="Matched Heatmap (GEN = GT)",
        filename=f"{output_prefix}_matched_heatmap.png",
        cmap="RdBu_r",
        vmin=0,
        vmax=1,
        special_zero_gray=True,
        colorbar_label="Correct Matrix Counter"
    )

    # Mismatch Heatmap
    _plot_heatmap(
        mismatch_matrix,
        title="Mismatch Heatmap (GEN ≠ GT)",
        filename=f"{output_prefix}_mismatch_heatmap.png",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        special_zero_gray=True,
        colorbar_label="Incorrect Matrix Value"
    )


    # Uncertainty Heatmap
    _plot_heatmap(
        uncertainty_matrix,
        title="Uncertainty Heatmap (GEN == 0)",
        filename=f"{output_prefix}_uncertainty_heatmap.png",
        cmap="YlOrBr",
        vmin=0,
        vmax=1,
        special_zero_gray=False,
        colorbar_label="Uncertainty Counter"
    )


# ---------------- Save Outputs ----------------
def save_all_outputs(filepath, labels, gen_clean, gt_clean, metrics_evaluation, output_prefix):
    """
    Save aligned DSM, evaluation metrics, and derived matrices to Excel. Saving heatmaps to PNG.
    """
    # Sheet 1: DSM
    df_dsm = pd.DataFrame(gen_clean, index=labels, columns=labels)

    # Sheet 2: Evaluation Metrics
    df_metrics = pd.DataFrame(list(metrics_evaluation.items()), columns=["Metric", "Value"])

    # Sheet 3: Matched DSM
    matched_matrix = pd.DataFrame("", index=labels, columns=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if gen_clean[i, j] == gt_clean[i, j]:
                matched_matrix.iloc[i, j] = 1

    # Sheet 4: Mismatch DSM
    mismatch_matrix = pd.DataFrame("", index=labels, columns=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if gen_clean[i, j] != gt_clean[i, j]:
                mismatch_matrix.iloc[i, j] = gen_clean[i, j]

    # Sheet 5: Uncertainty DSM
    uncertainty_matrix = pd.DataFrame("", index=labels, columns=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if gen_clean[i, j] == 0:
                uncertainty_matrix.iloc[i, j] = 1

    # Save to Excel workbook
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df_dsm.to_excel(writer, sheet_name="DSM")
        df_metrics.to_excel(writer, sheet_name="Evaluation Metrics", index=False)
        matched_matrix.to_excel(writer, sheet_name="Matched DSM")
        mismatch_matrix.to_excel(writer, sheet_name="Mismatch DSM")
        uncertainty_matrix.to_excel(writer, sheet_name="Uncertainty DSM")

    # Save heatmaps as PNG (Comment out if multirun execution is performed, decreasing execution time)
    generate_dsm_heatmaps(matched_matrix, mismatch_matrix, uncertainty_matrix, labels, output_prefix)


# ---------------- Main Execution ----------------
def run_evaluation_for_file(gen_path, index):
    """
    Running the evaluation for a single generated DSM file.
    """
    gt_df = pd.read_excel(GT_DSM_PATH, index_col=0)
    gen_df = pd.read_excel(gen_path, index_col=0)

    gt_elements, gen_elements, matched_elements, gt_aligned, gen_aligned = interactive_align_and_clean_dsms(gt_df, gen_df)

    if not matched_elements:
        print(f"[!] No matching components between GT and {os.path.basename(gen_path)}. Skipping.")
        return

    # Convert to int matrices
    gt_matrix = gt_aligned.values.astype(int)
    gen_matrix = gen_aligned.values.astype(int)

    # Compute metrics
    metrics_evaluation = calculate_evaluation_metrics(gt_matrix, gen_matrix, gt_elements, gen_elements)

    # Save outputs
    gt_name = os.path.splitext(os.path.basename(GT_DSM_PATH))[0]
    gen_name = os.path.splitext(os.path.basename(gen_path))[0]

    EXCEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'Evaluated_DSMs')
    FIGURE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'Evaluation_figures')
    os.makedirs(EXCEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

    output_file = os.path.join(EXCEL_OUTPUT_DIR, f"{gen_name}_{gt_name}_evaluated.xlsx")
    prefix = os.path.join(FIGURE_OUTPUT_DIR, f"{gen_name}_{gt_name}")


    save_all_outputs(
    filepath=output_file,
    labels=matched_elements,
    gen_clean=gen_matrix,
    gt_clean=gt_matrix,
    metrics_evaluation=metrics_evaluation,
    output_prefix=prefix
    )

    print(f"DSM {index} results saved to: {os.path.basename(output_file)}")

# Find all generated DSM files
def find_gen_files():
    files = []
    for fname in os.listdir(GEN_DSM_DIR):
        for pattern in GEN_PATTERNS:
            if fname.startswith(pattern) and fname.endswith(".xlsx"):
                files.append(os.path.join(GEN_DSM_DIR, fname))
    return sorted(files)

# Main execution iterating over generated DSM files
if __name__ == '__main__':
    gen_files = find_gen_files()
    for index, gen_path in enumerate(gen_files):
        run_evaluation_for_file(gen_path, index)
