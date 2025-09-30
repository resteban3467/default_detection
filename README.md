# Predicción de Riesgo de no pago de Tarjetas de Crédito

## Resumen del Proyecto
Este proyecto de Machine Learning tiene como objetivo desarrollar y evaluar un modelo capaz de predecir la probabilidad de que un cliente de tarjeta de crédito no pague el próximo mes. Utilizando un dataset público de un banco, se exploraron, pre-procesaron y modelaron los datos, comparando algoritmos como **XGBoost** y **Random Forest**. El flujo de trabajo incluye la optimización de hiperparámetros con **Optuna/GridSearchCV** y la interpretación del modelo final mediante **SHAP** para entender los factores de riesgo clave.

---
## Tabla de Contenidos
* [Problemática de Negocio](#problemática-de-negocio)
* [Dataset](#dataset)
* [Stack Tecnológico](#stack-tecnológico)
* [Metodología](#metodología)
* [Resultados Clave](#resultados-clave)
* [Cómo Ejecutar el Proyecto](#cómo-ejecutar-el-proyecto)
* [Autor](#autor)
* [Agradecimientos y fuentes](#agradecimientos-y-fuentes)

---
## Problemática de Negocio
La morosidad en los pagos de tarjetas de crédito representa una pérdida financiera significativa para los bancos. Se busca construir una herramienta de soporte a la decisión que permita a la entidad financiera:
1.  **Identificar** a los clientes con alta probabilidad de default.
2.  **Optimizar las estrategias de gestión de riesgo**, como ajustar límites de crédito.
3.  **Reducir las pérdidas** asociadas al incumplimiento de pagos.

---
## Dataset
El conjunto de datos utilizado es **"Default of Credit Card Clients Dataset"** del repositorio de la UCI. Contiene información demográfica y el historial de pagos y facturación de 30,000 clientes en Taiwán de abril a septiembre de 2005.

---
## Stack Tecnológico
* **Lenguajes:** `Python`, `SQL (PostgreSQL)`
* **Librerías de Datos:** `Pandas`, `NumPy`
* **Visualización:** `Matplotlib`, `Seaborn`, `Plotly`
* **Machine Learning:** `Scikit-learn`, `XGBoost`
* **Optimización:** `Optuna` / `GridSearchCV`
* **Explicabilidad:** `SHAP`
* **Entorno:** `Jupyter Notebooks`, `Git`

---
## Metodología

El proyecto siguió un ciclo de vida de ciencia de datos estructurado:

1.  **Ingesta y Preprocesamiento:** Los datos se cargaron en una base de datos PostgreSQL. Se utilizó SQL para realizar la limpieza inicial y la **ingeniería de características**, creando ratios de utilización de crédito y de pago.
2.  **Análisis Exploratorio de Datos (EDA):** Se analizaron las distribuciones, correlaciones y se identificó el **desbalance de clases** (78% No Default vs. 22% Default), el cual era considerable, pero no problemático.
3.  **Modelado y Optimización:**
    * Se probaron dos modelos de ensamble: `XGBoost` y `RandomForestClassifier`.
    * Se manejó el desbalance de clases usando los parámetros `scale_pos_weight` y `class_weight='balanced'`.
    * Se optimizaron los hiperparámetros maximizando la métrica **ROC AUC**.
4.  **Evaluación y Calibración:** Se compararon los modelos usando Precision, Recall y F1-Score. Se demostró que el Random Forest ofrecía el mejor balance inicial. Posteriormente, se **ajustó el umbral de decisión** para encontrar el punto óptimo entre la detección de morosos (Recall) y la reducción de falsas alarmas (Precision).
5.  **Explicabilidad:** Se utilizó la librería **SHAP** sobre el modelo campeón para interpretar sus decisiones, identificando las características más influyentes tanto a nivel global como para predicciones individuales.

---
## Resultados Clave

* **Modelo Campeón:** El `RandomForestClassifier` optimizado, con un umbral de decisión de **0.55**, fue seleccionado como el mejor modelo.
* **Métricas Finales (Clase 1: Default):**
    * **Precisión:** 0.50
    * **Recall:** 0.56
    * **F1-Score:** 0.53

Este modelo es el más equilibrado, logrando identificar a más de la mitad de los clientes riesgosos (56%) y asegurando que la mitad de sus alertas de riesgo son acertadas (50%).

* **Factores de Riesgo Más Importantes:** El análisis con SHAP reveló que el factor más predictivo es, por lejos, el **historial de pagos recientes ("PAY_0")**.

![Gráfico de barras](shap_feature_importance.png)

---
## Cómo Ejecutar el Proyecto

1.  **Clonar el repositorio:**
    ```bash
    git clone [URL-DE-TU-REPOSITORIO]
    cd [default_detection]
    ```
2.  **Crear el entorno virtual e instalar dependencias:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
3.  **Ejecutar los notebooks:** Abre Jupyter Lab (`jupyter lab`) y ejecuta los notebooks en orden numérico.

---
## Autor
* **[Esteban Rojas]**
* **LinkedIn:** www.linkedin.com/in/esteban-rojas-millar
* **Email:** [resteban3467@gmail.com]

---
## Agradecimientos y Fuentes

Este proyecto fue posible gracias al uso de datos públicos de alta calidad. Se extienden los agradecimientos tanto a los creadores originales del dataset como al repositorio que lo mantiene accesible.

### Fuente Original del Dataset
El dataset fue creado y donado originalmente por los siguientes investigadores:

> Yeh, I-Cheng; Lien, Che-Hui (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*.

### Repositorio de Datos
El dataset fue obtenido del Repositorio de Machine Learning de la UCI, mantenido por la Universidad de California, Irvine.

> Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.