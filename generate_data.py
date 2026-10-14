import pandas as pd 
import numpy as np 
np.random.seed(42)
n = 500 
data = pd.DataFrame({
"age": np.random.randint(18, 70, n), # âge entre 18 et 70 ans
"salaire": np.random.randint(1200, 8000, n), # salaire mensuel simulé
"abonnement": np.random.choice(["Basique", "Standard", "Premium"], n), # type d'offre
"anciennete": np.random.randint(0, 15, n), # ancienneté en années
"consommation": np.random.randint(0, 100, n), # score d'usage du

})

data["churn"] = (
(data["consommation"] < 30) &
(data["anciennete"] < 2)
).astype(int) 
data.to_csv("churn_data.csv", index=False)

print(data.head())
print(f"Taux de churn : {data['churn'].mean():.1%}")