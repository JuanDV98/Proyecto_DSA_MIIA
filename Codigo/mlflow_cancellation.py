# 0. Importar librerías
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, accuracy_score
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# 1. Cargar y preparar datos
df = pd.read_parquet('filtered_base.parquet')
df.rename(columns={'NBS_mORA': 'NBS_mora'}, inplace=True)
y = df['Estado_Final_2_meses_despues']
X = df.drop(columns=['Estado_Final_2_meses_despues'])

# 2. Definir columnas numéricas y categóricas
num_cols = ['dias_mora', 'Ant_pol', 'NBS_mora', 'Edad', 'total_mora', 'Total_Activas', 'NBS_Vigente']
cat_cols = ['Plan_Agrupado', 'MedioPago', 'genero']

# 3. Configurar preprocesamiento
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# 4. Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 5. Calcular scale_pos_weight para XGBoost (para manejar desbalance de clases)
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# 6. Configurar experimento en MLflow
mlflow.set_tracking_uri("http://13.218.234.21:8050")  # Cambia si es otra IP
experiment = mlflow.set_experiment("model-comparison-cancellation")

# 7. Importar modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# 8. Lista de modelos a evaluar
modelos = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, class_weight='balanced', random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42),
    "XGBoost": xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        colsample_bytree=1.0,
        learning_rate=0.1,
        max_depth=7,
        subsample=0.6,
        random_state=42
    )
}

# 9. Entrenar y registrar cada modelo
for nombre_modelo, modelo in modelos.items():
    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=nombre_modelo):
        # Crear pipeline con preprocesamiento + modelo
        pipe = Pipeline([("prep", preprocess), ("clf", modelo)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        # Registrar parámetros relevantes
        if nombre_modelo == "XGBoost":
            mlflow.log_param("scale_pos_weight", scale_pos_weight)
            mlflow.log_param("max_depth", 7)
            mlflow.log_param("subsample", 0.6)
            mlflow.log_param("learning_rate", 0.1)
        elif nombre_modelo == "RandomForest":
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 8)
        elif nombre_modelo == "AdaBoost":
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("learning_rate", 0.5)
        elif nombre_modelo == "LogisticRegression":
            mlflow.log_param("max_iter", 1000)

        # Calcular métricas
        recall    = recall_score(y_test, preds, pos_label=1)
        precision = precision_score(y_test, preds, pos_label=1)
        f1        = f1_score(y_test, preds, pos_label=1)
        accuracy  = accuracy_score(y_test, preds)

        # Registrar métricas
        mlflow.log_metric("recall_cancelled", recall)
        mlflow.log_metric("precision_cancelled", precision)
        mlflow.log_metric("f1_cancelled", f1)
        mlflow.log_metric("accuracy", accuracy)

        # Firmar el modelo y registrar
        input_example = X_train.iloc[:1]
        signature = infer_signature(X_train, pipe.predict(X_train))

        mlflow.sklearn.log_model(
            pipe,
            artifact_path=f"{nombre_modelo}-pipeline",
            input_example=input_example,
            signature=signature
        )

        # Imprimir reporte en consola
        print(f"\n==== {nombre_modelo} ====")
        print(classification_report(y_test, preds, target_names=["Active (0)", "Cancelled (1)"], digits=3))