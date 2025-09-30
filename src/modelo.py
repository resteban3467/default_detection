#!/usr/bin/env python
# coding: utf-8

# # Importar librerías y módulos relevantes

# In[1]:


# Esenciales
import pandas as pd
import numpy as np
# Visualización
import seaborn as sns
import matplotlib.pyplot as plt
# ML y Modelado
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
import xgboost as xgb
import optuna
# Explicabilidad del Modelo
import shap
# Almacenar el Modelo
import pickle
from pathlib import Path


# # Cargar datos desde SQL

# In[2]:


df = pd.read_csv("../data/procesado/data_feature_engineering.csv")


# # Revisión exploratoria de los datos

# ## Confirmación de que los datos de extrajeron correctamente desde SQL

# In[3]:


df.head()


# In[4]:


df.shape


# # Visualización Exploratoria
# Primero, usaré un gráfico de barras para ver rápidamente la proporción entre los datos que son declarados en default y los que no. Por otro lado, usaré un mapa de calor para revisar las correlaciones entre las características de la base de datos.

# In[5]:


plt.figure(figsize=(8, 6))
sns.countplot(x='default_payment_next_month', data=df)
plt.title('Distribución de Clientes (0: No Default, 1: Default)')
plt.show()

# 2. Mapa de calor de correlaciones
plt.figure(figsize=(20, 15))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Mapa de Calor de Correlaciones')
plt.show()


# Se puede apreciar que hay un desbalance, si bien no es muy extremo, es lo suficientemente destacable como para tomarlo en cuenta para los parámetros que se usarán en los modelos.
# 
# Además, se puede ver que en el mapa de calor hay una fuerte correlación entre las características del grupo "PAY" y el grupo "BILL". Pero esto cobrará relevancia más adelante, en las explicaciones posteriores.

# # Definición de variables X e Y

# In[6]:


X = df.copy()
X = X.drop(columns = {"default_payment_next_month"}, axis = 1)
y = df["default_payment_next_month"]


# # Separación entre grupos de entrenamiento y pruebas

# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    stratify = y, 
                                                    test_size = 0.25, 
                                                    random_state = 20)


# # Preparación de entrenamiento con Optuna
# 
# He optado por usar optuna para entrenar el modelo XGBoost, el principal motivo es que es relativamente rápido y eficiente, ya que su trabajo es "aprender" de los resultados anteriores y ver qué parametros son más relevantes para obtener los mejores resultados. En lo que respecta a su aprendizaje, tendrá como tarea maximizar la métrica de "**roc_auc**", esto tendrá más relevancia más adelante.

# In[10]:


def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'verbosity': 0,
        'n_jobs': -1,
        'random_state': 20,
        'tree_method': 'hist',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
    }


    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    param['scale_pos_weight'] = scale_pos_weight


    model = xgb.XGBClassifier(**param)

    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)

    return np.mean(score)

study = optuna.create_study(direction='maximize')

study.optimize(objective, n_trials=100, timeout=600)

best_params = study.best_trial.params


# ## Visualización de su aprendizaje
# 
# El modelo trabaja en base a varios intentos en los que busca parámetros de forma aleatoria, pero que de a poco aprende cuál es la combinación que da mejores resultados en base a la tarea que le dí. Dicho esto, se puede aprecia que al principio solía dar resultados más bien azarosos, pero conforme fue aprendiendo, sus intentos se iban volviendo más y más constantes.

# In[11]:


optuna.visualization.plot_optimization_history(study)


# ## Preparación del modelo optimizado
# Luego de determinar la mejor combinación de parámetros, se guardará esa combinación y se pondrá a prueba.

# In[13]:


final_model = xgb.XGBClassifier(**best_params)


# In[14]:


final_model.fit(X_train, y_train)


# # Evaluación Modelo XGBoost

# In[15]:


y_pred = final_model.predict(X_test)
y_pred_proba = final_model.predict_proba(X_test)[:, 1]
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión')
plt.show()

print("\nReporte de Clasificación")
print(classification_report(y_test, y_pred))


# ## Explicación de métricas
# "Precision" se usa para medir la proporción entre los verdaderos positivos y todas las predicciones de positivos (verdaderos positivos y falsos positivos). Dicho esto, nos dice que el 67% de las veces el modelo predecirá que un cliente iría a default y realmente se irá a default. Por lo tanto, más de la mitad de las veces se presentarán casos en los que el modelo determinará que un cliente irá a default y sí habrá sido así realmente.
# 
# En cambio, "Recall" se usa para medir la proporción entre verdaderos positivos y todas las predicciones que deberían ser positivos (verdaderos positivos y falsos negativos), siendo en este caso un 36%, por lo tanto, el modelo es más bien ineficiente a la hora de detectar los clientes que realmente se irán a default, lo cual es bastante importante.
# 
# Finalmente, "f1_score" sirve para tener una idea general del rendimiento del modelo, esto se debe a que es la media harmónica entre las métricas recién mencionadas, sin embargo, creo que es más importante para este caso concentrarnos en las dos anteriores.

# ## Correción de umbral
# 
# Ya que la precisión es considerablemente más alta que el recall, puede ser buena idea cambiar un cierto **umbral**, lo relevante de este umbral es que me permite modificar el grado de permisividad que tendrá el modelo. El valor base es 0.5, pero si lo bajo podría ser menos riguroso con los valores que predice como positivo, pero a cambio sería más riguroso con los valores que deberían ser verdaderos.
# 
# El motivo por el cual se escogió la métrica "roc_auc", es porque es una forma de decir que busque un modelo que funcione bien sin importar el umbral, para así poder definirlo posteriormente y ver cuál sería el óptimo según el caso.

# In[31]:


y_pred_proba = final_model.predict_proba(X_test)[:, 1]

nuevo_umbral = 0.26

y_pred_nuevo = (y_pred_proba >= nuevo_umbral).astype(int)

print(f"Resultados con umbral de {nuevo_umbral}")
print(classification_report(y_test, y_pred_nuevo))


# ## Umbral óptimo
# 
# Después de experimentar un poco con valores, determiné que lo más ideal es aumentar el valor de recall, ya que es más importante detectar la mayor cantidad de personas posible que realmente no pagarán su próxima cuota. Sin embargo, tampoco es buena idea dejar un valor de precisión demasiado bajo, porque de ese modo saldrían demasiadas predicciones de clientes que no pagarán cuando realmente si iban a pagar. Considero que 50% es lo mínimo que debería de tener un modelo en precisión.

# ## Explicación del modelo (SHAP)
# Para evitar los problemas de explicabilidad que conlleva usar XGBoost, usaré SHAP para buscar una mejor explicación acerca de las columnas más relevantes para el modelo. 
# 
# En primer lugar, usaré un gráfico "Beeswarm" para mostrar el impacto de las características en la predicción de cada observación.
# 
# Finalmente, usaré un gráfico de barras para mostrar la importancia de cada característica de forma global.

# # Nuevo Modelo
# Ya que no se han podido obtener resultados satisfactorios, es momento de probar con otro modelo y comparar. En este caso, optaré por usar un **Random Forest Classifier**.

# ## Importar módulos

# In[17]:


from sklearnex import patch_sklearn # Solo importar esto si se tiene un procesador Intel
patch_sklearn() # Si no se cumple el requisito mencionado, entonces es mejor borrar esta sección.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV # Usaré esta búsqueda de parámetros porque el parche recién instalado acelera muchísimo el código, de otro modo, sería buena idea usar otra forma de mejorar los parámetros.


# ## Definición de parámetros

# In[18]:


parámetros = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 4],
    "max_features": ["sqrt", "log2"]
}


# ## Instanciar modelo

# In[19]:


rf_classifier = RandomForestClassifier(
    class_weight='balanced',
    random_state=20,
    n_jobs=-1
)

grid_search_rf = GridSearchCV(estimator=rf_classifier,
                              param_grid=parámetros,
                              scoring='roc_auc', 
                              cv=5,               
                              verbose=1,
                              n_jobs=-1)


# ## Entrenar el modelo

# In[20]:


grid_search_rf.fit(X_train, y_train)


# ## Evaluar el modelo
# 
# Para evaluar el modelo, aplicaremos los mismos pasos que con el anterior, una matriz de confusión y luego una evaluación de sus métricas "precision", "recall" y "f1_score".

# In[21]:


y_pred_rf = grid_search_rf.predict(X_test)

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Default', 'Default'],
            yticklabels=['No Default', 'Default'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión - Random Forest')
plt.show()

print(classification_report(y_test, y_pred_rf))


# # Experimentar con un nuevo umbral
# 
# La precisión es un poco baja y recall está bien, pero si subo un poco el umbral podría obtener un resultado más equilibrado.

# In[22]:


y_pred_proba_rf = grid_search_rf.predict_proba(X_test)[:, 1]

nuevo_umbral = 0.55

y_pred_nuevo_rf = (y_pred_proba_rf >= nuevo_umbral).astype(int)

print(f"Resultados con umbral de {nuevo_umbral}")

print(classification_report(y_test, y_pred_nuevo_rf))


# ## Explicación del nuevo modelo
# 
# Usaré los mismos gráficos -*beeswarm y barras*- para explicar el modelo.

# In[23]:


mejor_modelo = grid_search_rf.best_estimator_


# In[24]:


explainer = shap.TreeExplainer(mejor_modelo)

shap_values = explainer(X_test)

shap_values_class_1 = shap_values[..., 1]

shap.summary_plot(shap_values_class_1, X_test, plot_type='dot', show=False)
plt.title("Impacto de las Features en la Predicción de 'Default'")
plt.show()

shap.summary_plot(shap_values_class_1, X_test, plot_type='bar', show=False)
plt.title("Importancia Global de las Features para Predecir 'Default'")

fig = plt.gcf()
fig.axes[-1].set_xlabel("Impacto Promedio en la Salida del Modelo (Valor SHAP Absoluto)")
plt.tight_layout()
plt.show()


# # Impacto de variables
# 
# Según lo visto en el mapa de calor, las características que comparten una alta correlación están con un nivel de importancia más bien similar, esto es normal, ya que el modelo tenderá a darles la misma importancia predictiva. Sin embargo, cabe destacar que las características "PAY_0" y "PAY_2" son demasiado destacadas por encima del resto como para confundirlas. Por lo tanto, el foco de interés debiese caerles encima sin importar lo mencionado previamente.

# # Guardar modelo
# Ya que efectivamente se pudo obtener un resultado más balanceado, es importante guardar el modelo.

# In[25]:


project_root = Path.cwd().parent

carp_pkl = project_root / "pkl"

carp_pkl.mkdir(exist_ok=True, parents = True)

mejor_modelo_path = carp_pkl / "mejor_modelo_rf.pkl"

grid_search_path = carp_pkl / "grid_search_rf_resultados.pkl"

with open(mejor_modelo_path, 'wb') as f:
        pickle.dump(mejor_modelo, f)

with open(grid_search_path, 'wb') as f:
        pickle.dump(grid_search_rf, f)


# # Conclusión
# 
# En este proyecto:
# - Revisamos el desbalance de los datos.
# - Visualizamos las características e investigamos las relaciones (correlación) entre las características.
# - Después revisamos dos modelos de Machine Learning para ver cuál sería más apropiado para detectar el default de los clientes.
# - Se concluyó que el más apropiado sería el de Random Forest Classifier, ya que obtuvo los resultados más balanceados y que permite detectar la mayor cantidad de defaults que de verdad lo irían a ser, sin necesidad de caer demasiado en falsos positivos.
