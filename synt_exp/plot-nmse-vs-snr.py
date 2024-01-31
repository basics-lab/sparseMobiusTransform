import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 504)
pd.set_option('display.width', 1000)

sys.path.append("..")
sys.path.append("../src")

from pathlib import Path

if __name__ == '__main__':

    exp_dir = Path(f"results/debug-90f80a04")

    nmse_type = "nmse_mobius"

    results_df = pd.read_pickle(exp_dir / "result.pkl")
    group_by = ["method", "n", "q", "num_subsample", "num_repeat", "b", "noise_sd", "sparsity"]

    results_df[nmse_type] = results_df[nmse_type].clip(upper=1)

    means = results_df.groupby(group_by, as_index=False).mean()
    stds = results_df.groupby(group_by, as_index=False).std()

    _, ax = plt.subplots(1, 1, figsize=(3.5, 3), dpi=300)

    sparsity_list = np.unique(results_df["sparsity"])

    snr_values = [[] for _ in range(len(sparsity_list))]

    for i in means.index:
        mean_row = means.iloc[i]
        s_idx = np.where(mean_row["sparsity"] == sparsity_list)[0][0]
        snr_values[s_idx].append((mean_row["snr"], mean_row[nmse_type]))

    for s in range(len(sparsity_list)):
        ax.plot(*zip(*snr_values[s]), "o-", label=f"S = {sparsity_list[s]}", markersize=4)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('NMSE')
    ax.legend()
    ax.grid()

    (exp_dir / "figs").mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(exp_dir / f'figs/nmse-vs-snr.pdf', bbox_inches='tight',
                transparent="True", pad_inches=0)

    plt.show()


