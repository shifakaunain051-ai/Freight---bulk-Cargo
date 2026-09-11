import json

with open("scratch/walk_forward_results.json") as f:
    res = json.load(f)

for r in res:
    if "Base_v3" in r["Feature_Set"]:
        folds_str = ", ".join([f"{m:.3f}" for m in r["Fold_MAEs"]])
        print(f"{r['Algorithm']:22} | {r['Feature_Set']:18} | {r['Mean_MAE']:8.4f} | {r['Worst_Fold_MAE']:6.4f} | {r['Dir_Acc_%']:5.1f}% | [{folds_str}]")
