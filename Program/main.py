from data_loader import load_data
from preprocessing import preprocess_data
from models import get_models
from evaluation import evaluate_model
from interpretation import (
    linear_regression_coefficients,
    gradient_boosting_importance
)
from visualization import (
    plot_model_comparison,
    plot_feature_importance,
    plot_predictions
)
import os


def main():
    csv_path = "../podskup podataka.csv"

    # --------------------------------------------------
    # 1. Učitavanje podataka
    # --------------------------------------------------
    df = load_data(csv_path)

    # --------------------------------------------------
    # 2. Ciljna varijabla
    # --------------------------------------------------
    target_column = "consumption_normalized"

    # --------------------------------------------------
    # 3. Pretprocesiranje
    # --------------------------------------------------
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        df=df,
        target_column=target_column
    )

    feature_names = df.drop(columns=[target_column]).columns

    # --------------------------------------------------
    # 4. Modeli
    # --------------------------------------------------
    models = get_models()
    results = {}

    # direktorij za grafove
    os.makedirs("results", exist_ok=True)

    print("\nREZULTATI MODELA")
    print("=" * 60)

    # --------------------------------------------------
    # 5. Treniranje, evaluacija i interpretacija
    # --------------------------------------------------
    for name, model in models.items():
        print(f"\nMODEL: {name}")
        print("-" * 60)

        # Treniranje
        model.fit(X_train, y_train)

        # Evaluacija
        mae, rmse = evaluate_model(model, X_test, y_test)
        results[name] = {"MAE": mae, "RMSE": rmse}

        print(f"MAE : {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")

        # --------------------------------------------------
        # Interpretacija
        # --------------------------------------------------
        if name == "Linear Regression":
            print("\nNajutjecajnije varijable (Linear Regression):")
            coef_df = linear_regression_coefficients(
                model,
                feature_names
            )
            print(coef_df.to_string(index=False))

        elif name == "Gradient Boosting":
            print("\nNajvažnije značajke (Gradient Boosting):")
            importance_df = gradient_boosting_importance(
                model,
                feature_names
            )
            print(importance_df.to_string(index=False))

            # Graf važnosti značajki
            plot_feature_importance(
                importance_df,
                save_path="results/feature_importance_gradient_boosting.png"
            )

            # Scatter: stvarno vs. predviđeno
            y_pred = model.predict(X_test)
            plot_predictions(
                y_test,
                y_pred,
                name,
                save_path="results/predicted_vs_actual_gradient_boosting.png"
            )

    # --------------------------------------------------
    # 6. Graf usporedbe modela
    # --------------------------------------------------
    plot_model_comparison(
        results,
        save_path="results/model_comparison.png"
    )

    print("\nAnaliza završena.")
    print("Grafovi su spremljeni u folder 'results/'.")


if __name__ == "__main__":
    main()