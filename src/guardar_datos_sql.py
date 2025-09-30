#!/usr/bin/env python
# coding: utf-8

# # Importar librerías
# 
# Para poblar el servidor SQL, se usará este medio por lo tanto, hay que leer el archivo csv a través de pandas y el resto se verá en la otra plataforma.

# In[1]:


import pandas as pd
from sqlalchemy import create_engine, text


# # Transformar de csv a DataFrame
# Este paso es muy importante para poder leer los datos en el archivo original y poder llevarlos a SQL.

# In[2]:


import pandas as pd
ruta = "../data/raw/UCI_Credit_Card.csv"
df = pd.read_csv(ruta)


# # Exploración inicial
# Primero, hay que revisar que la tabla corresponde a lo indicado en la documentación de kaggle.

# In[3]:


df.dtypes


# Se puede apreciar que todo está en orden, sin embargo, es un poco incómodo ver que la última columna cuenta con "." como delimitadores, por lo tanto, se hará un pequeño cambio para sustituirlo por un "_".

# In[4]:


df = df.rename(columns = {"default.payment.next.month": "default_payment_next_month"})
df.dtypes


# # Cargar credenciales
# 
# Por motivos de privacidad, he optado por usar el método de la celda de abajo para acceder a mi postgresql y no revelarlos, sin embargo, no es necesario para nada en otros dispositivos, uno solo puede simplemente agregar los datos correspondientes a las variables definidas.

# In[5]:


import os
from dotenv import load_dotenv

load_dotenv()

db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')


# In[7]:


engine = create_engine(f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}')


# In[14]:


with engine.connect() as connection:
    with connection.begin() as transaction:
        try:
            df.to_sql(
                'tarj_cred_default',
                con=connection,
                if_exists='append',
                index=False
            )
            transaction.commit()
            print("Datos cargados")
        except Exception as e:
            print(f"Ocurrió un error: {e}")
            transaction.rollback() 


# # Código para trabajar en SQL
# 
# ```SQl
# SELECT
#     "LIMIT_BAL",
#     "SEX",
#     "EDUCATION",
#     "MARRIAGE",
#     "AGE",
#     "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
#     "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
#     "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
# 
#     CASE WHEN "LIMIT_BAL" > 0 THEN "BILL_AMT1" / "LIMIT_BAL" ELSE 0 END AS "ratio_utilización_1",
#     CASE WHEN "LIMIT_BAL" > 0 THEN "BILL_AMT2" / "LIMIT_BAL" ELSE 0 END AS "ratio_utilización_2",
#     CASE WHEN "LIMIT_BAL" > 0 THEN "BILL_AMT3" / "LIMIT_BAL" ELSE 0 END AS "ratio_utilización_3",
#     CASE WHEN "LIMIT_BAL" > 0 THEN "BILL_AMT4" / "LIMIT_BAL" ELSE 0 END AS "ratio_utilización_4",
#     CASE WHEN "LIMIT_BAL" > 0 THEN "BILL_AMT5" / "LIMIT_BAL" ELSE 0 END AS "ratio_utilización_5",
#     CASE WHEN "LIMIT_BAL" > 0 THEN "BILL_AMT6" / "LIMIT_BAL" ELSE 0 END AS "ratio_utilización_6",
#     CASE WHEN "BILL_AMT2" > 0 THEN "PAY_AMT1" / "BILL_AMT2" ELSE 0 END AS "ratio_pago_1",
#     CASE WHEN "BILL_AMT3" > 0 THEN "PAY_AMT2" / "BILL_AMT3" ELSE 0 END AS "ratio_pago_2",
#     CASE WHEN "BILL_AMT4" > 0 THEN "PAY_AMT3" / "BILL_AMT4" ELSE 0 END AS "ratio_pago_3",
#     CASE WHEN "BILL_AMT5" > 0 THEN "PAY_AMT4" / "BILL_AMT5" ELSE 0 END AS "ratio_pago_4",
#     CASE WHEN "BILL_AMT6" > 0 THEN "PAY_AMT5" / "BILL_AMT6" ELSE 0 END AS "ratio_pago_5",
# 
#     "default_payment_next_month"
# FROM
#     tarj_cred_default;

# Se dejó la columna de sexo porque no es ético diferenciar a los clientes por el sexo con el que nacieron, de la misma forma que no se consideran sus nombres. 
# Por otro lado, se crearon nuevas columnas que surgen de la interacción de otras, siendo los grupos de "ratio de utilización" y los "ratio de pago". En primer lugar, los ratios de utilización son la proporción entre la cantidad de dinero usada como crédito y el límite establecido. Por su parte, el grupo de ratio de pago sirve para ver la proporción entre la cantidad pagada y la deuda correspondiente al mes siguiente.
