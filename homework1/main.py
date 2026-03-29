from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 2026
TRAIN_RATIO = 0.8
DATA_PATH = Path("Concrete_Data_Yeh.csv")
OUTPUT_DIR = Path("outputs")


def load_data():
    df = pd.read_csv(DATA_PATH)
    return split_data(df, shuffle=True)


def split_data(df, shuffle):
    features = df.iloc[:, :-1].copy()
    target = df.iloc[:, -1].copy()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        train_size=TRAIN_RATIO,
        test_size=1 - TRAIN_RATIO,
        random_state=RANDOM_STATE,
        shuffle=shuffle,
    )
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    return x_train, x_test, y_train, y_test


def fill_missing_values(x_train, x_test):
    fill_values = x_train.median()
    return x_train.fillna(fill_values), x_test.fillna(fill_values)


def build_models():
    return {
        "linear_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "pca_linear_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95)),
                ("model", LinearRegression()),
            ]
        ),
        "mlp_small": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(32,),
                        activation="relu",
                        solver="adam",
                        learning_rate_init=0.01,
                        max_iter=3000,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "mlp_deep": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32, 16),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate_init=0.003,
                        max_iter=4000,
                        early_stopping=True,
                        validation_fraction=0.1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def train_predict_with_target_scaling(model, x_train, y_train, x_test):
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    model.fit(x_train, y_train_scaled)
    pred_scaled = model.predict(x_test).reshape(-1, 1)
    return y_scaler.inverse_transform(pred_scaled).ravel()


def select_correlated_features(x_train, y_train, top_k=6):
    corr_series = x_train.copy()
    corr_series["target"] = y_train.values
    corr_series = corr_series.corr()["target"].drop("target")
    corr_series = corr_series.loc[corr_series.abs().sort_values(ascending=False).index]
    selected_features = corr_series.index[:top_k].tolist()
    return selected_features, corr_series


def evaluate_predictions(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def save_prediction_plot(name, y_true, y_pred, metrics):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.75, edgecolors="none")
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    plt.plot([low, high], [low, high], linestyle="--", linewidth=2, color="tab:red")
    plt.xlabel("True Strength (MPa)")
    plt.ylabel("Predicted Strength (MPa)")
    plt.title(f"{name} Prediction Performance")
    plt.text(
        0.05,
        0.95,
        f"MSE={metrics['mse']:.3f}\nRMSE={metrics['rmse']:.3f}\nMAE={metrics['mae']:.3f}\nR2={metrics['r2']:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_prediction.png", dpi=200)
    plt.close()


def save_correlation_plot(corr_series, selected_features):
    colors = ["#C44E52" if feature in selected_features else "#4C72B0" for feature in corr_series.index]
    plt.figure(figsize=(9, 4.8))
    plt.bar(corr_series.index, corr_series.values, color=colors)
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(rotation=25)
    plt.ylabel("Correlation with strength")
    plt.title("Feature Correlation Analysis")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_correlation.png", dpi=200)
    plt.close()


def save_correlation_heatmap(df):
    corr_matrix = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr_matrix.columns)
    ax.set_title("Correlation Heatmap")

    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()


def save_shuffle_vs_no_shuffle_distribution_plot(df, x_train, x_test, y_train, y_test):
    split_idx = int(len(df) * TRAIN_RATIO)
    target_col = df.columns[-1]
    no_shuffle_train = df.iloc[:split_idx]
    no_shuffle_test = df.iloc[split_idx:]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].hist(no_shuffle_train["age"], bins=20, alpha=0.7, density=True, label="Train", color="#4C72B0")
    axes[0, 0].hist(no_shuffle_test["age"], bins=20, alpha=0.7, density=True, label="Test", color="#C44E52")
    axes[0, 0].set_title("Age Distribution Without Shuffling")
    axes[0, 0].set_xlabel("Age (day)")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()

    axes[0, 1].hist(no_shuffle_train[target_col], bins=20, alpha=0.7, density=True, label="Train", color="#55A868")
    axes[0, 1].hist(no_shuffle_test[target_col], bins=20, alpha=0.7, density=True, label="Test", color="#8172B3")
    axes[0, 1].set_title("Strength Distribution Without Shuffling")
    axes[0, 1].set_xlabel("Concrete Strength (MPa)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()

    axes[1, 0].hist(x_train["age"], bins=20, alpha=0.7, density=True, label="Train", color="#4C72B0")
    axes[1, 0].hist(x_test["age"], bins=20, alpha=0.7, density=True, label="Test", color="#C44E52")
    axes[1, 0].set_title("Age Distribution With Shuffling")
    axes[1, 0].set_xlabel("Age (day)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()

    axes[1, 1].hist(y_train, bins=20, alpha=0.7, density=True, label="Train", color="#55A868")
    axes[1, 1].hist(y_test, bins=20, alpha=0.7, density=True, label="Test", color="#8172B3")
    axes[1, 1].set_title("Strength Distribution With Shuffling")
    axes[1, 1].set_xlabel("Concrete Strength (MPa)")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shuffle_vs_no_shuffle_distribution.png", dpi=200)
    plt.close()


def save_mlp_loss_plot(name, model):
    mlp = model.named_steps["model"]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(np.arange(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, color="#4C72B0", linewidth=2)
    ax.set_title(f"{name} Training Loss Curve")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_loss_curve.png", dpi=200)
    plt.close()


def save_pca_plot(x_train, y_train):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    pca = PCA(n_components=2)
    components = pca.fit_transform(x_train_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    scatter = axes[0].scatter(
        components[:, 0],
        components[:, 1],
        c=y_train,
        cmap="viridis",
        s=20,
        alpha=0.8,
    )
    axes[0].set_title("PCA Projection of Training Data")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    fig.colorbar(scatter, ax=axes[0], label="Strength (MPa)")

    explained = np.cumsum(pca.fit(StandardScaler().fit_transform(x_train)).explained_variance_ratio_)
    axes[1].plot(np.arange(1, len(explained) + 1), explained, marker="o", color="#C44E52")
    axes[1].axhline(0.95, linestyle="--", color="#55A868", linewidth=2)
    axes[1].set_title("Cumulative Explained Variance")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Explained Variance Ratio")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pca_analysis.png", dpi=200)
    plt.close()


def evaluate_no_shuffle_mlp(df):
    x_train, x_test, y_train, y_test = split_data(df, shuffle=False)
    x_train, x_test = fill_missing_values(x_train, x_test)
    models = build_models()
    results = []

    for name in ("mlp_small", "mlp_deep"):
        model = models[name]
        predictions = train_predict_with_target_scaling(model, x_train, y_train, x_test)
        metrics = evaluate_predictions(y_test, predictions)
        results.append({"model": name, "split": "no_shuffle", **metrics})

    return pd.DataFrame(results)


def save_shuffle_comparison_plot(results_df, no_shuffle_results_df):
    shuffled = results_df[results_df["model"].isin(["mlp_small", "mlp_deep"])].copy()
    shuffled["split"] = "shuffle"
    combined = pd.concat(
        [shuffled[["model", "split", "rmse", "r2"]], no_shuffle_results_df[["model", "split", "rmse", "r2"]]],
        ignore_index=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    model_order = ["mlp_small", "mlp_deep"]
    x = np.arange(len(model_order))
    width = 0.34

    for idx, split_name in enumerate(["shuffle", "no_shuffle"]):
        subset = combined[combined["split"] == split_name].set_index("model").loc[model_order]
        offset = -width / 2 if split_name == "shuffle" else width / 2
        axes[0].bar(x + offset, subset["rmse"], width=width, label=split_name, alpha=0.85)
        axes[1].bar(x + offset, subset["r2"], width=width, label=split_name, alpha=0.85)

    axes[0].set_title("MLP RMSE: Shuffle vs No Shuffle")
    axes[0].set_ylabel("RMSE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_order)
    axes[0].legend()

    axes[1].set_title("MLP R2: Shuffle vs No Shuffle")
    axes[1].set_ylabel("R2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_order)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "mlp_shuffle_comparison.png", dpi=200)
    plt.close()


def save_summary_plot(results_df):
    ordered = results_df.sort_values("mse", ascending=True)
    x = np.arange(len(ordered))
    width = 0.24

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    bars1 = ax1.bar(x - width, ordered["mse"], width=width, label="MSE", color="#4C72B0")
    bars2 = ax1.bar(x, ordered["rmse"], width=width, label="RMSE", color="#55A868")
    bars3 = ax1.bar(x + width, ordered["mae"], width=width, label="MAE", color="#8172B3")
    ax1.set_ylabel("Error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ordered["model"], rotation=15)

    ax2 = ax1.twinx()
    line = ax2.plot(x, ordered["r2"], color="#C44E52", marker="o", linewidth=2, label="R2")[0]
    ax2.set_ylabel("R2 Score")
    ax2.set_ylim(0, min(1.05, max(0.1, ordered["r2"].max() + 0.1)))

    for bar in list(bars1) + list(bars2) + list(bars3):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.legend([bars1, bars2, bars3, line], ["MSE", "RMSE", "MAE", "R2"], loc="upper center", ncol=4)
    plt.title("Model Comparison on Concrete Strength Prediction")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=200)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    x_train, x_test, y_train, y_test = load_data()
    x_train, x_test = fill_missing_values(x_train, x_test)
    models = build_models()
    correlated_features, corr_series = select_correlated_features(x_train, y_train, top_k=6)
    models["correlation_linear_regression"] = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    results = []
    save_correlation_plot(corr_series, correlated_features)
    save_correlation_heatmap(df)
    save_shuffle_vs_no_shuffle_distribution_plot(df, x_train, x_test, y_train, y_test)
    save_pca_plot(x_train, y_train)
    (OUTPUT_DIR / "selected_features.txt").write_text(
        "Selected features based on absolute correlation:\n"
        + "\n".join(correlated_features)
        + "\n",
        encoding="utf-8",
    )

    for name, model in models.items():
        current_x_train = x_train
        current_x_test = x_test
        if name == "correlation_linear_regression":
            current_x_train = x_train[correlated_features]
            current_x_test = x_test[correlated_features]

        if name.startswith("mlp"):
            predictions = train_predict_with_target_scaling(
                model, current_x_train, y_train, current_x_test
            )
            save_mlp_loss_plot(name, model)
        else:
            model.fit(current_x_train, y_train)
            predictions = model.predict(current_x_test)
        metrics = evaluate_predictions(y_test, predictions)
        save_prediction_plot(name, y_test, predictions, metrics)

        result = {"model": name, **metrics}
        if "pca" in model.named_steps:
            result["pca_components"] = int(model.named_steps["pca"].n_components_)
        if name == "correlation_linear_regression":
            result["selected_features"] = ",".join(correlated_features)
        if "model" in model.named_steps and hasattr(model.named_steps["model"], "n_iter_"):
            result["iterations"] = int(model.named_steps["model"].n_iter_)
        results.append(result)

    results_df = pd.DataFrame(results).sort_values("mse", ascending=True).reset_index(drop=True)
    no_shuffle_results_df = evaluate_no_shuffle_mlp(df)
    results_df.to_csv(OUTPUT_DIR / "results.csv", index=False)
    no_shuffle_results_df.to_csv(OUTPUT_DIR / "no_shuffle_mlp_results.csv", index=False)
    save_summary_plot(results_df)
    save_shuffle_comparison_plot(results_df, no_shuffle_results_df)

    print("Finished experiments. Results:")
    print(results_df.to_string(index=False))
    print("\nMLP results without shuffling:")
    print(no_shuffle_results_df.to_string(index=False))
    print(f"\nSaved figures and tables to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
