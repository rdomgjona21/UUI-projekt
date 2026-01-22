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

    df = load_data(csv_path)
    target_column = "consumption_normalized"

    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        df=df,
        target_column=target_column
    )

    feature_names = df.drop(columns=[target_column]).columns

    models = get_models()
    results = {}

    os.makedirs("results", exist_ok=True)

    print("\nREZULTATI MODELA")
    print("=" * 60)

    for name, model in models.items():
        print(f"\nMODEL: {name}")
        print("-" * 60)


        model.fit(X_train, y_train)


        mae, rmse = evaluate_model(model, X_test, y_test)
        results[name] = {"MAE": mae, "RMSE": rmse}

        print(f"MAE : {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")


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

            plot_feature_importance(
                importance_df,
                save_path="results/feature_importance_gradient_boosting.png"
            )

            y_pred = model.predict(X_test)
            plot_predictions(
                y_test,
                y_pred,
                name,
                save_path="results/predicted_vs_actual_gradient_boosting.png"
            )


    plot_model_comparison(
        results,
        save_path="results/model_comparison.png"
    )

    print("\nAnaliza završena.")
    print("Grafovi su spremljeni u folder 'results/'.")


if __name__ == "__main__":
    main()