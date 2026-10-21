import mlflow

# Récupérer les runs, triés par F1 décroissant
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    order_by=["metrics.f1 DESC"],
)

# Colonnes à afficher
colonnes = [
    "tags.mlflow.runName",
    "params.n_estimators",
    "params.max_depth",
    "metrics.accuracy",
    "metrics.f1",
]

if runs.empty:
    print("Aucun run trouvé dans l'expérience churn-prediction.")
else:
    print(runs[colonnes].to_string(index=False))

    # La première ligne correspond au meilleur score F1
    meilleur = runs.iloc[0]

    print()
    print(
        "Meilleur run :", meilleur["tags.mlflow.runName"],
        "| id :", meilleur["run_id"][:8] + "...",
    )