import os
import re

model_names = {
    'sarimax_diario.py': 'sarimax',
    'tar_diario.py': 'tar',
    'ridge_diario.py': 'ridge',
    'rf_diario.py': 'rf',
    'ml_diario.py': 'lightgbm',
    'lstm_diario.py': 'lstm',
    'gru_diario.py': 'gru'
}

for fname, model_name in model_names.items():
    path = os.path.join('src', fname)
    with open(path, 'r') as f:
        content = f.read()

    # 1. Add PRED_DIR
    if "PRED_DIR = OUTPUT_DIR / 'predictions'" not in content:
        content = re.sub(
            r"(OUTPUT_DIR\.mkdir\(exist_ok=True\)|OUTPUT_DIR\.mkdir\(parents=True, exist_ok=True\))",
            r"\1\nPRED_DIR = OUTPUT_DIR / 'predictions'\nPRED_DIR.mkdir(parents=True, exist_ok=True)",
            content
        )

    # 2. Add DataFrame saving
    date_var = "test_pld.index" if fname == "sarimax_diario.py" else "X_test.index"
    save_code = f"""
        pd.DataFrame({{
            "data": {date_var}.strftime("%Y-%m-%d"),
            "real": y_true,
            "previsto": y_pred
        }}).to_csv(PRED_DIR / f"{model_name}_{{submercado}}_fold{{fold+1}}.csv", sep=";", index=False)
"""

    if "PRED_DIR /" not in content.split("fold_mae =")[0]:
        content = re.sub(
            r"(\s+fold_mae = mean_absolute_error\(y_true, y_pred\))",
            save_code + r"\1",
            content
        )

    with open(path, 'w') as f:
        f.write(content)
