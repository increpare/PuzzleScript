import json
import os

import numpy as np
import pandas as pd


seed = 12
max_gen_attempts = 20

sweep_dir = os.path.join('logs', f'sweep-{seed}')
with open(os.path.join(sweep_dir, 'stats_and_dists.json')) as f:
    vanilla_stats = json.load(f)

with open(os.path.join(sweep_dir, 'fromIdea', 'stats_and_dists.json')) as f:
    stats_from_idea = json.load(f)

agg_stats = {}
for from_idea in [0, 1]:
    for cot in [0, 1]:
        fewshot = 1
        key = ("fromIdea-1_" if from_idea else "") + f"fewshot-{fewshot}_cot-{cot}"
        first_comps = []
        first_solves = []
        min_edit_dists = []
        skipped = 0
        comps = 0
        solves = 0
        if from_idea:
            stats = stats_from_idea
        else:
            stats = vanilla_stats
        for stat in stats[key]:
            if stat['skipped']:
                skipped += 1
                continue
            compiled_iters = stat['compiledIters']
            solved_iters = stat['solvedIters']
            if compiled_iters:
                comps += 1
                first_comp = min(compiled_iters)
                # Only consider diversity metric if the game actually compiles
                min_edit_dists.append(stat['min_dist'])
            else:
                first_comp = max_gen_attempts
            if solved_iters:
                solves += 1
                first_solve = min(solved_iters)
            else:
                first_solve = max_gen_attempts
            first_comps.append(first_comp)
            first_solves.append(first_solve)
        mean_first_comp = np.mean(first_comps)
        std_first_comp = np.std(first_comps)
        mean_first_solve = np.mean(first_solves)
        std_first_solve = np.std(first_solves)
        pct_comp = comps / len(stats[key])
        pct_solve = solves / len(stats[key])
        agg_stats[key] = {
            'mean_first_comp': mean_first_comp,
            'std_first_comp': std_first_comp,
            'mean_first_solve': mean_first_solve,
            'std_first_solve': std_first_solve,
            'pct_comp': pct_comp,
            'pct_solve': pct_solve,
            'mean_edit_dist': np.mean(min_edit_dists),
            'std_edit_dist': np.std(min_edit_dists),
            'skipped': skipped,
        }

# Now make a pandas dataframe out of this
# Convert to DataFrame
df = pd.DataFrame(agg_stats).T

# Extract hierarchical indices
# index_tuples = [
#     (bool(int(fewshot)), bool(int(cot)))
#     for fewshot, cot in df.index.str.extract(r'fewshot-(\d+)_cot-(\d+)').values
# ]
index_tuples = []
for key in df.index:
    from_idea = key.startswith("fromIdea-1")
    cot = "cot-1" in key
    index_tuples.append((bool(int(from_idea)), bool(int(cot))))
df.index = pd.MultiIndex.from_tuples(index_tuples, names=['From Idea', 'Chain of Thought'])

# Format mean columns to include std as "+/-" values
df["First Compile"] = df.apply(
    lambda row: f"{row['mean_first_comp']:.1f} ± {row['std_first_comp']:.1f}", axis=1
)
df["First Solve"] = df.apply(
    lambda row: f"{row['mean_first_solve']:.1f} ± {row['std_first_solve']:.1f}", axis=1
)
df["Min Edit Dist"] = df.apply(
    lambda row: f"{row['mean_edit_dist']:.1f} ± {row['std_edit_dist']:.1f}", axis=1
)

# Drop original columns
df = df.drop(columns=[
    'mean_first_comp', 'std_first_comp', 'mean_first_solve', 'std_first_solve', 'mean_edit_dist', 'std_edit_dist', 'skipped'
])

# Bold the least values in "First Compile" and "First Solve" columns
for col in ["First Compile", "First Solve"]:
    min_value = df[col].apply(lambda x: float(x.split(" ± ")[0])).min()
    df[col] = df[col].apply(
        lambda x: f"\\textbf{{{x}}}" if float(x.split(" ± ")[0]) == min_value else x
    )

# Bold the greatest values in "pct_comp" and "pct_solve" columns
for col in ["pct_comp", "pct_solve"]:
    # Format as percentage, but escape the percent sign
    df[col] = df[col].apply(lambda x: f"{x:.0%}".replace("%", "\\%"))
    max_value = df[col].max()
    df[col] = df[col].apply(lambda x: f"\\textbf{{{x}}}" if x == max_value else x)

for col in ["Min Edit Dist"]:
    max_value = df[col].apply(lambda x: float(x.split(" ± ")[0])).max()
    df[col] = df[col].apply(
        lambda x: f"\\textbf{{{x}}}" if float(x.split(" ± ")[0]) == max_value else x
    )

# Rename columns to remove underscores
df = df.rename(columns=lambda x: x.replace("_", " ").title())

df = df.sort_index()

# Save DataFrame to LaTeX
# latex_path = "/mnt/data/modified_hierarchical_dataframe.tex"
latex_path = os.path.join(sweep_dir, 'fromIdea', 'agg_stats.tex')
df.to_latex(latex_path, escape=False, multirow=True)
# df.style.to_latex(latex_path, multirow_align='c', hrules=True, clines='all;data')


print(json.dumps(agg_stats, indent=4))
