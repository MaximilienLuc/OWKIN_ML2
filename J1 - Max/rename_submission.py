import nbformat as nbf
from pathlib import Path

nb_path = Path("baseline_max.ipynb")
nb = nbf.read(nb_path, as_version=4)

change_made = False
target_string = 'submission.to_csv(data_dir / "benchmark_test_output.csv", index=None)'
replacement_string = 'submission.to_csv(data_dir / "rf_optimized_optuna.csv", index=None)'

for cell in nb.cells:
    if cell.cell_type == "code":
        if target_string in cell.source:
            cell.source = cell.source.replace(target_string, replacement_string)
            change_made = True
            print("Renamed output file in notebook.")
            break

if change_made:
    nbf.write(nb, nb_path)
    print("Notebook updated successfully.")
else:
    print("Target string not found or already updated.")
