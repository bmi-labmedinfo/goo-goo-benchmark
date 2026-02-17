################################################################################
# Tabular Transformer benchmark on Medical data
################################################################################

import yaml
import numpy as np
import pandas as pd

from collections import Counter

from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
)
from sklearn.utils import shuffle

from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion

from tabdpt import TabDPTClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from tabicl import TabICLClassifier
import matplotlib.pyplot as plt
# -------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------

def get_dataset_info(name: str):
    for ds in config["datasets"]:
        if ds["name"] == name:
            return ds
    raise ValueError(f"Dataset '{name}' not found.")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Crea un preprocessor che gestisce automaticamente numeriche e categoriche."""
    cat_cols = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # dense output per TabPFN / TabNet / TabICL
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def prepare_X_y(dataset_info):
    """Carica il dataset, applica group_class se presente e mappa le classi in 0..K-1."""
    df = pd.read_csv(dataset_info["path"])
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")] # in some dataset we have the colum index
    target_col = dataset_info["target"]

    y_raw = df[target_col]

    # 1) Eventuale raggruppamento delle classi (es. per binarizzare)
    if "group_class" in dataset_info:
        # group_class è una stringa tipo "1,2,3" oppure "C,D"
        positive_tokens = [
            t.strip() for t in str(dataset_info["group_class"]).split(",") if t.strip()
        ]
        # Confronto sulle stringhe, così funziona sia per target numerici sia stringa
        y_str = y_raw.astype(str)
        y_bin = np.array(
            [1 if val in positive_tokens else 0 for val in y_str],
            dtype=int,
        )
        y_mapped = y_bin
    else:
        # mappa comunque tutte le classi in 0..K-1 per sicurezza (TabDPT-friendly)
        values = y_raw.to_numpy()
        unique_vals = pd.unique(values)
        class_to_int = {cls: i for i, cls in enumerate(unique_vals)}
        y_mapped = np.array([class_to_int[v] for v in values], dtype=int)

    X = df.drop(columns=target_col)

    return X, y_mapped


def evaluate_model(model, X_test, y_test, model_name):
    """Valuta in modo automatico binary vs multi-class."""
    y_pred = model.predict(X_test)
    result = {
        "model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "BalancedAcc": balanced_accuracy_score(y_test, y_pred),
    }

    n_classes = len(np.unique(y_test))

    # F1
    if n_classes == 2:
        result["F1"] = f1_score(y_test, y_pred)
    else:
        result["F1"] = f1_score(y_test, y_pred, average="macro")

    # AUC, se il modello espone predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                # binary
                result["AUC"] = roc_auc_score(y_test, y_prob[:, 1])
            else:
                # multi-class
                result["AUC"] = roc_auc_score(
                    y_test, y_prob, multi_class="ovr", average="macro"
                )
        except Exception:
            # se per qualche motivo l'AUC fallisce (classe unica in test, ecc.)
            result["AUC"] = np.nan
    else:
        result["AUC"] = np.nan

    return result


# -------------------------------------------------------------------------
# Caricamento config YAML
# -------------------------------------------------------------------------

with open("/Users/utente/Desktop/Tabular_Model/dataset_info.yaml", "r") as f:
    config = yaml.safe_load(f)

# Puoi scegliere qui il dataset
dataset_names = [
    "myocardial_infarction", # Funge
    "cdc_diabetes", # Funge
    "cyrrosis", # Funge ho messo C e Cl come stessa classe chiedere a googola se ha senso
    "Thyroid_Diff", # funge
    "glioma",# Funge
    "hepatitis",#funge
    "parkinson", #funge
    "Gh_TO",
    "Gh_T1",
    "Gh_T2",
    "student_depression",
    "Thyroid_cancer",
]
dataset_name = dataset_names[2]   # cambia indice per cambiare dataset

dataset_info = get_dataset_info(dataset_name)

print("######################## Dataset Info ###################################")
print(dataset_info)
print("#########################################################################")

# Prepara X, y
X, y = prepare_X_y(dataset_info)

print("Number of patients:", X.shape[0])
print("Number of features:", X.shape[1])
print("Class counts:", Counter(y))
print("Classes:", np.unique(y))

# -------------------------------------------------------------------------
# Modelli base + grid
# -------------------------------------------------------------------------

rf = RandomForestClassifier(random_state=42, class_weight="balanced")
rf_grid = {
    "n_estimators": [100, 300],
    "max_depth": [None, 5, 10],
    "min_samples_leaf": [1, 3, 5],
}

lasso = LogisticRegression(
    solver="saga",
    penalty="l1",
    max_iter=10000,
    random_state=42,
    class_weight="balanced",
)

lasso_grid = {
    "C": np.logspace(-3, 1, 5),
}

# -------------------------------------------------------------------------
# CV + modelli
# -------------------------------------------------------------------------

results = []

# Frazioni di dataset (in termini di dimensione del training set) da testare
fractions = [0.10, 0.20, 0.50, 0.75, 1.0]

n_splits = 5
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

np.random.seed(1)


X_sub = X.reset_index(drop=True)
y_sub = np.array(y)

for frac in fractions:
    print("\n" + "=" * 80)
    print(f" FRAZIONE DI TRAINING UTILIZZATA: {int(frac * 100)}%")
    print("=" * 80)

    for i, (train_index, test_index) in enumerate(kf.split(X_sub, y_sub)):
        print(f"\n################ Fold {i} (train fraction = {frac:.2f}) ################")

        # Split di base
        X_train_full = X_sub.iloc[train_index, :]
        y_train_full = y_sub[train_index]

        X_test = X_sub.iloc[test_index, :]
        y_test = y_sub[test_index]

        # Sotto-campionamento del training set secondo la frazione richiesta
        if frac < 1.0:
            sss = StratifiedShuffleSplit(
                n_splits=1, train_size=frac, random_state=42
            )
            sub_idx, _ = next(sss.split(X_train_full, y_train_full))
            X_train = X_train_full.iloc[sub_idx, :]
            y_train = y_train_full[sub_idx]
        else:
            X_train = X_train_full
            y_train = y_train_full

        print("Train size:", X_train.shape[0], "- Test size:", X_test.shape[0])

        # ---------------------------------------------------------------------
        # 1) Preprocessing UNICO per TUTTI i modelli
        # ---------------------------------------------------------------------
        preprocessor = build_preprocessor(X_train)

        # matrice numerica densa condivisa da tutti i modelli
        X_train_tab = preprocessor.fit_transform(X_train)
        X_test_tab = preprocessor.transform(X_test)

        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        # ---------------------------------------------------------------------
        # 2) Modelli ML classici (RF, LASSO)
        # ---------------------------------------------------------------------

        # ---- Random Forest ----
        try:
            rf_search = GridSearchCV(
                rf,
                rf_grid,
                cv=inner_cv,
                scoring="balanced_accuracy",
                n_jobs=-1,
            )
            rf_search.fit(X_train_tab, y_train)
            res_rf = evaluate_model(
                rf_search.best_estimator_, X_test_tab, y_test, "RandomForest"
            )
            res_rf.update(
                {
                    "fraction": frac,
                    "fold": i,
                    "dataset": dataset_name,
                }
            )
            results.append(res_rf)
            print("RF done.")
        except Exception as e:
            print(f"[WARNING] RandomForest failed on fold {i}: {e}")

        # ---- LASSO Logistic Regression ----
        try:
            lasso_search = GridSearchCV(
                lasso,
                lasso_grid,
                cv=inner_cv,
                scoring="balanced_accuracy",
                n_jobs=-1,
            )
            lasso_search.fit(X_train_tab, y_train)
            res_lasso = evaluate_model(
                lasso_search.best_estimator_, X_test_tab, y_test, "LASSO_Logistic"
            )
            res_lasso.update(
                {
                    "fraction": frac,
                    "fold": i,
                    "dataset": dataset_name,
                }
            )
            results.append(res_lasso)
            print("LASSO done.")
        except Exception as e:
            print(f"[WARNING] LASSO failed on fold {i}: {e}")

        # ---------------------------------------------------------------------
        # 3) Modelli Tabular Transformer
        # ---------------------------------------------------------------------

        # ---- TabPFN v2.5 raw ----
        try:
            tabpfn_class = TabPFNClassifier()
            tabpfn_class.fit(X_train_tab, y_train)
            res_tabpfn = evaluate_model(
                tabpfn_class, X_test_tab, y_test, "TabPFN_2.5_raw"
            )
            res_tabpfn.update(
                {
                    "fraction": frac,
                    "fold": i,
                    "dataset": dataset_name,
                }
            )
            results.append(res_tabpfn)
            print("TabPFN 2.5 raw done.")
        except Exception as e:
            print(f"[WARNING] TabPFN_2.5_raw failed on fold {i}: {e}")

        # ---- TabPFN v2.5 tuning ----
        try:
            tabpfn_class_t = TabPFNClassifier(
                tuning_config={"tune_decision_thresholds": True}
            )
            tabpfn_class_t.fit(X_train_tab, y_train)
            res_tabpfn_t = evaluate_model(
                tabpfn_class_t, X_test_tab, y_test, "TabPFN_2.5_tuning"
            )
            res_tabpfn_t.update(
                {
                    "fraction": frac,
                    "fold": i,
                    "dataset": dataset_name,
                }
            )
            results.append(res_tabpfn_t)
            print("TabPFN 2.5 tuning done.")
        except Exception as e:
            print(f"[WARNING] TabPFN_2.5_tuning failed on fold {i}: {e}")

        # ---- TabPFN v2 raw ----
        try:
            tabpfn_2_class = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2
            )
            tabpfn_2_class.fit(X_train_tab, y_train)
            res_tabpfn2 = evaluate_model(
                tabpfn_2_class, X_test_tab, y_test, "TabPFN_2_raw"
            )
            res_tabpfn2.update(
                {
                    "fraction": frac,
                    "fold": i,
                    "dataset": dataset_name,
                }
            )
            results.append(res_tabpfn2)
            print("TabPFN v2 raw done.")
        except Exception as e:
            print(f"[WARNING] TabPFN_2_raw failed on fold {i}: {e}")

        # ---- TabPFN v2 tuning ----
        try:
            tabpfn_2_class_t = TabPFNClassifier.create_default_for_version(
                ModelVersion.V2,
                tuning_config={"tune_decision_thresholds": True},
            )
            tabpfn_2_class_t.fit(X_train_tab, y_train)
            res_tabpfn2_t = evaluate_model(
                tabpfn_2_class_t, X_test_tab, y_test, "TabPFN_2_tuning"
            )
            res_tabpfn2_t.update(
                {
                    "fraction": frac,
                    "fold": i,
                    "dataset": dataset_name,
                }
            )
            results.append(res_tabpfn2_t)
            print("TabPFN v2 tuning done.")
        except Exception as e:
            print(f"[WARNING] TabPFN_2_tuning failed on fold {i}: {e}")

        # ---- TabDPT ----
        try:
            tabdpt_class = TabDPTClassifier()
            tabdpt_class.fit(X_train_tab, y_train)
            res_tabdpt = evaluate_model(
                tabdpt_class, X_test_tab, y_test, "TabDPT_raw"
            )
            res_tabdpt.update(
                {
                    "fraction": frac,
                    "fold": i,
                    "dataset": dataset_name,
                }
            )
            results.append(res_tabdpt)
            print("TabDPT done.")
        except Exception as e:
            print(f"[WARNING] TabDPT failed on fold {i}: {e}")

        # ---- TabNet ----
        try:
            tabnet_class = TabNetClassifier()
            tabnet_class.fit(X_train_tab, y_train)
            res_tabnet = evaluate_model(
                tabnet_class, X_test_tab, y_test, "TabNet_raw"
            )
            res_tabnet.update(
                {
                    "fraction": frac,
                    "fold": i,
                    "dataset": dataset_name,
                }
            )
            results.append(res_tabnet)
            print("TabNet done.")
        except Exception as e:
            print(f"[WARNING] TabNet failed on fold {i}: {e}")

        # ---- TabICL ----
        try:
            tabicl_class = TabICLClassifier()
            tabicl_class.fit(X_train_tab, y_train)
            res_tabicl = evaluate_model(
                tabicl_class, X_test_tab, y_test, "TabICL_raw"
            )
            res_tabicl.update(
                {
                    "fraction": frac,
                    "fold": i,
                    "dataset": dataset_name,
                }
            )
            results.append(res_tabicl)
            print("TabICL done.")
        except Exception as e:
            print(f"[WARNING] TabICL failed on fold {i}: {e}")

# -------------------------------------------------------------------------
# Risultati
# -------------------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n================ PRIME RIGHE RISULTATI =================")
print(results_df.head())

# Salva TUTTI i risultati per-fold in CSV
perfold_csv = f"{dataset_name}_learning_curves_perfold.csv"
results_df.to_csv(perfold_csv, index=False)
print(f"\n[INFO] Risultati per-fold salvati in: {perfold_csv}")

# Riassunto mean ± std per (fraction, model)
summary = (
    results_df.groupby(["fraction", "model"])[
        ["Accuracy", "BalancedAcc", "F1", "AUC"]
    ]
    .agg(["mean", "std"])
    .round(3)
)

print("\n================ SUMMARY PER FRAZIONE E MODELLO =================")
print(summary)

# -------------------------------------------------------------------------
# Grafici: andamento delle performance al crescere del dataset
# -------------------------------------------------------------------------

metrics_to_plot = ["Accuracy", "BalancedAcc", "F1", "AUC"]
fractions_sorted = sorted(results_df["fraction"].unique())

for metric in metrics_to_plot:
    plt.figure(figsize=(8, 6))

    for model_name in results_df["model"].unique():
        df_plot = (
            results_df.groupby(["fraction", "model"])[metric]
            .mean()
            .reset_index()
        )
        df_plot = df_plot[df_plot["model"] == model_name].sort_values("fraction")

        # se la metrica è tutta NaN per questo modello, skippa
        if df_plot[metric].notna().sum() == 0:
            continue

        plt.plot(
            df_plot["fraction"],
            df_plot[metric],
            marker="o",
            label=model_name,
        )

    plt.xticks(
        fractions_sorted,
        [f"{int(f * 100)}%" for f in fractions_sorted],
    )
    plt.xlabel("Frazione del training set")
    plt.ylabel(metric)
    plt.title(
        f"Andamento di {metric} al variare della dimensione del dataset ({dataset_name})"
    )
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(title="Modello", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"learning_curve_{dataset_name}_{metric}.png", dpi=300)
    plt.show()
