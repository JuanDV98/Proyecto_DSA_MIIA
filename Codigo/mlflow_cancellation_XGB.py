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

# 3. Configurar Preprocesamiento
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# 4. Split y entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 5. Calculate scale_pos_weight based on training data to handle class imbalance
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# 6. Configurar experimento en MLflow
experiment = mlflow.set_experiment("xgboost-cancellation")

# 7. Iniciar el registro del experimento
with mlflow.start_run(experiment_id=experiment.experiment_id):
    # 7.1 Parámetros del modelo (pueden ser ajustadas para diferentes corridas)
    colsample_bytree = 1.0   # Proporción de variables (features) usadas en cada árbol (1.0 = todas)
    learning_rate = 0.1      # Cuánto aporta cada árbol nuevo al modelo final (velocidad de aprendizaje)
    max_depth = 7            # Profundidad máxima de cada árbol de decisión (controla complejidad)
    subsample = 0.6          # Proporción de muestras de entrenamiento usadas en cada árbol (0.6 = 60%)

    # 7.2 Configurar modelo XGBoost
    model = xgb.XGBClassifier(
        objective = "binary:logistic",
        eval_metric = "aucpr",
        scale_pos_weight = scale_pos_weight,
        colsample_bytree = colsample_bytree,
        learning_rate = learning_rate,
        max_depth = max_depth,
        subsample = subsample,
        random_state = 42
    )

    # 7.3 Crear pipeline ensamblado con preprocesamienot y modelo XGBoost
    pipe = Pipeline([("prep", preprocess), ("clf", model)])

    # 7.4 Entrenamiento del modelo
    pipe.fit(X_train, y_train)

    # 7.5 Realizar predicciones
    preds = pipe.predict(X_test)

    # 7.6 Log parameters
    mlflow.log_param("colsample_bytree", colsample_bytree)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("subsample", subsample)

    # 7.7 Calcular métricas
    recall    = recall_score(y_test, preds, pos_label=1)
    precision = precision_score(y_test, preds, pos_label=1)
    f1        = f1_score(y_test, preds, pos_label=1)
    accuracy  = accuracy_score(y_test, preds)

    # 7.8 Log metrics
    mlflow.log_metric("recall_cancelled", recall)
    mlflow.log_metric("precision_cancelled", precision)
    mlflow.log_metric("f1_cancelled", f1)
    mlflow.log_metric("accuracy", accuracy)

    # 7.9 Create input example and signature
    input_example = X_train.iloc[:1]
    signature = infer_signature(X_train, pipe.predict(X_train))

    # 7.10 Log the model
    mlflow.sklearn.log_model(
        pipe,
        artifact_path="xgboost-pipeline",
        input_example=input_example,
        signature=signature
    )

    # Print classification report (no graphs as specified)
    print("\nXGBOOST – Classification Report (threshold = 0.50)\n")
    print(classification_report(y_test, preds, target_names=["Active (0)", "Cancelled (1)"], digits=3))