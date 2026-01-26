import matplotlib.pyplot as plt


def plot_model_comparison(results, save_path=None):

    models = list(results.keys())
    mae_values = [results[m]["MAE"] for m in models]
    rmse_values = [results[m]["RMSE"] for m in models]

    x = range(len(models))

    plt.figure(figsize=(8, 5))
    plt.bar(x, mae_values, label="MAE")
    plt.bar(x, rmse_values, bottom=mae_values, label="RMSE")

    plt.xticks(x, models, rotation=20)
    plt.ylabel("Pogreška")
    plt.title("Usporedba pogreške predikcije po modelima")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def plot_feature_importance(df, save_path=None):
    plt.figure(figsize=(8, 5))
    plt.barh(df["Feature"], df["Importance"])
    plt.xlabel("Važnost")
    plt.title("Važnost značajki (Gradient Boosting)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def plot_predictions(y_test, y_pred, model_name, save_path=None):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Stvarna vrijednost")
    plt.ylabel("Predviđena vrijednost")
    plt.title(f"Stvarno vs. predviđeno ({model_name})")

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()