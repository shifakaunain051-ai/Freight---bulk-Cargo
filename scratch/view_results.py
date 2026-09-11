import json

with open("scratch/walk_forward_results.json") as f:
    res = json.load(f)

print(f"{'Algorithm':22} | {'Feature Set':18} | {'Mean MAE':8} | {'Worst':6} | {'DirAcc':6} | Fold MAEs (F1 to F5)")
print("-" * 95)
for r in res:
    if r["Mean_MAE"] < 0.60:
        folds_str = ", ".join([f"{m:.3f}" for m in r["Fold_MAEs"]])
        print(f"{r['Algorithm']:22} | {r['Feature_Set']:18} | {r['Mean_MAE']:8.4f} | {r['Worst_Fold_MAE']:6.4f} | {r['Dir_Acc_%']:5.1f}% | [{folds_str}]")
