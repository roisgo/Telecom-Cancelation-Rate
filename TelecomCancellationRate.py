import pandas as pd
import numpy as np
import re
import os, sys
sys.path.append(os.getcwd()) 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score



try:
    df_contract = pd.read_csv('datasets/final_provider/contract.csv')
    df_personal = pd.read_csv('datasets/final_provider/personal.csv')
    df_internet = pd.read_csv('datasets/final_provider/internet.csv')
    df_phone = pd.read_csv('datasets/final_provider/phone.csv')                

except:
    df_contract = pd.read_csv('/datasets/final_provider/contract.csv')
    df_personal = pd.read_csv('/datasets/final_provider/personal.csv')
    df_internet = pd.read_csv('/datasets/final_provider/internet.csv')
    df_phone = pd.read_csv('/datasets/final_provider/phone.csv')


# Combinar todo el DataFrames en un solo DataFrame
df = pd.merge(df_contract, df_personal, on='customerID', how='outer')
df = pd.merge(df, df_internet, on='customerID', how='outer')
df = pd.merge(df, df_phone, on='customerID', how='outer')

# Eliminando la columna customerID porque ya no lo vamos a necesitar
df = df.drop('customerID', axis=1).reset_index(drop=True)


# Vamos a trabajar sobre nuestra variable objetivo realizando algunas acciones

# Reemplazar todos los valores 'No' en la columna EndDate, sustituyendolo por None
df['EndDate'] = np.where(df['EndDate'] == 'No', None, df['EndDate'])

# Cambiar el formato de columnas con fechas 
df['BeginDate'] = pd.to_datetime(df['BeginDate'], format='%Y-%m-%d')
df['EndDate'] = pd.to_datetime(df['EndDate'], format='%Y-%m-%d')

# Creacion de una columna que nos proporcione mayor información para calcular el numero de meses en la compañia
df['MonthsInCompany'] = (df['EndDate'] - df['BeginDate']) / pd.Timedelta(days=30)

# Reemplazar todos los valores NaN con la última fecha en la columna EndDate restandole el valor de la columna BeginDate 
df['MonthsInCompany'].fillna((df['EndDate'].max() - df['BeginDate']) / pd.Timedelta(days=30), inplace=True)

# Creacion de una columna para albergar a la variable objetivo que nos indica si el cliente a dejado la compañia o no.
df['Exited'] = df['EndDate'].notna().astype('uint8')

# Eliminanmos las columnas de fecha que ya no son de utilidad
df.drop(['BeginDate', 'EndDate'], axis=1, inplace=True)


# En virtud de que algunos modelos no aceptan nombres de columnas con Mayúsculas y minúsculas y/o espacios entre los nombres, procederemos a unificar la visualización de los nombres de las columnas
# 

# %%
def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

df.columns = [to_snake_case(col) for col in df.columns]

# %%
df.head(5)

# %%
df.info()

# %% [markdown]
# De esta informacion podemos decir que de la columna 11 a la 17 tenemos missing values, los podemos sustituir por "0" y convertir dichas columnas a valores numéricos. Esto debido a que todas estas columnas estan relacionadas con el servicio de internet y simplemente son clientes que no cuentan con el servicio.
# 

# %%
# Manejo de valores faltantes

# Rellena valores faltantes en 'MultipleLines' con 'No' debido a que no cuentan con multiples lineas
df['multiple_lines'] = df['multiple_lines'].fillna('No')  

# Convierte a numérico y manejar errores
df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')  

# Rellena valores NaN con 0
df['total_charges'] = df['total_charges'].fillna(0)  

for col in ["internet_service", "online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]:
  df[col] = df[col].fillna("No")

df['senior_citizen'] = df['senior_citizen'].astype('uint8')

# %%
df.info()

# %% [markdown]
# A partir de este momento tenemos un Data frame sin valaores nulos con 19 columnas, continuaremos con la revisión.
# 

# %%
# Procedemos a identificar y eliminar duplicados
print(f'El número total de filas duplicadas en este archivo es de {df.duplicated().sum()} filas.')


# %%
df.drop_duplicates(keep='last', inplace=True)

# %%
df.info()

# %%
#Vamos a explorar como se comporta nuestra variable objetivo
class_counts = df['exited'].value_counts()

print(class_counts)

sns.countplot(x='exited', data=df)
plt.show()

# %% [markdown]
# Claramente hay un desequilibiro de clases, situación que hay que tomar en cuenta al momento de desarrollar el modelo.
# 

# %% [markdown]
# Para responder a la pregunta de saber que tienen en comun los clientes que han abandonado la compañia, realizare la exploración con las dos variables que considero mas importantes "Total_charges" y "monthly_charges"
# 


# %%
# Creamos la grafica para visualizar la distribución de la variable MonthlyCharges
sns.displot(df['monthly_charges'], kde=True)
plt.show()

# %%
# Creamos la grafica para visualizar la distribución de la variable MonthlyCharges
sns.displot(df['total_charges'], kde=True)
plt.show()

# %% [markdown]
# Despues de analizar las dos graficas anteriores se observa que en ambas hay un elevado numero de clientes que han tenido cargos muy bajos tanto mensuales como totales, una primera idea que se me ocurre es que haya valores "outlayers" vamos a verificar esta suposicion con graficas boxplot
# 

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[['monthly_charges']])
plt.title('Boxplot de Cargos Mensuales')
plt.xlabel('Variable')
plt.ylabel('Valor')
plt.show()

# %%
plt.figure(figsize=(8, 6))
sns.boxplot(data=df[['total_charges']])
plt.title('Boxplot de Cargos Totales')
plt.xlabel('Variable')
plt.ylabel('Valor')
plt.show()

# %% [markdown]
# Se observa que no hay valores fuera de rango en ninguna de las dos columnas analizadas, por lo qu no es necesario eliminar outlayers
# 

# %% [markdown]
# Antes de proceder a revisar los modelos vamos a realizar un breve analisis de los factores en comun que pueden tener los clientes que han abandonado la compañia, a mi parecer pueden ser la informacion contenida en la columna monthy_charges, meses_contrato, type e internet.
# 

# %%
# plt.figure(figsize=(10, 6))
sns.countplot(x='type', hue='exited', data=df)
plt.title('Tasa de abandono por tipo de contrato')
plt.xlabel('Tipo de contrato')
plt.ylabel('Número de clientes')
plt.legend(title='Abandono')
plt.show()

# %% [markdown]
# Obsrvamos que los clientes que mas abandonan a la compañia son aquellos que tienen un cobro mensual y los que menos lo hacen son los que pagan cada dos años
# 

# %%
# plt.figure(figsize=(10, 8))
sns.countplot(x='payment_method', hue='exited', data=df)
plt.title('Tasa de abandono por método de pago')
plt.xlabel('Metodo de pago')
plt.ylabel('Número de clientes')
plt.legend(title='Abandono')
plt.show()

# %%
# plt.figure(figsize=(10, 8))
sns.countplot(x='internet_service', hue='exited', data=df)
plt.title('Tasa de abandono por servicio de internet')
plt.xlabel('Tipo de internet')
plt.ylabel('Número de clientes')
plt.legend(title='Abandono')
plt.show()

# %% [markdown]
# Se puede observa que los clientes que mas abandonan la empresa son los que tienen internet de fibra optica que tambien son los que tienen los costos mas elevados.Asimismo quienes realizan su pago con cheque electronico tambien son los que tiene la mayor taza de abandono comparado con los otros tres tipos de pagos.
# 
# ___
## %% [markdown]
# Observamos que tenemos muchas variables categoricas las cuales toman valores que no afectan significativamente a nuestra variable objetivo, por lo que resulta conveniente aplicarles un procesamiento OneHot encoder

# %%
columns_to_dummify = df.select_dtypes(include=['object']).columns.tolist()


# Aplicar get_dummies en un bucle for
for column in columns_to_dummify:
    df = pd.get_dummies(df, columns=[column], drop_first=True)

# %%
df.info()

# %% [markdown]
# Ahora que ya solo tenemos variables numéricas, procederemos a crear una matriz de correlación para visualizar la importancia de las variables en relación con nuestra variable objetivo.

# %%
# Calcular la matriz de correlación
correlation_matrix = df.corr()

# Seleccionar la fila o columna correspondiente a 'Exited'
correlation_with_exited = correlation_matrix['exited']  # Si 'Exited' es una columna
# O
correlation_with_exited = correlation_matrix.loc['exited']  # Si 'Exited' es una fila

# Ordenar los valores de correlación
sorted_correlation = correlation_with_exited.sort_values(ascending=False)

# Crear un nuevo mapa de calor solo con las variables ordenadas
sorted_correlation_matrix = correlation_matrix.loc[sorted_correlation.index, sorted_correlation.index]

# Crear la figura con un tamaño específico
plt.figure(figsize=(10, 8))

# Visualizar la matriz de correlación ordenada como un mapa de calor
sns.heatmap(sorted_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Agregar título
plt.title('Matriz de Correlación Ordenada con respecto a Exited')

# Mostrar el mapa de calor
plt.show()

# %%
# Convertir el diccionario a un DataFrame
df_resultados = pd.DataFrame(sorted_correlation_matrix)

# Imprimir el DataFrame
print(df_resultados)



# %%
# Especificar la variable objetivo
target_var = 'exited'

# Extraer la fila correspondiente a la variable objetivo
correlations = correlation_matrix[target_var].drop(target_var)

# Establecer un umbral de correlación
correlation_threshold = 0.15

#Filtrar variables cuya correlación con la variable objetivo esté por debajo del umbral
variables_to_keep = correlations[correlations.abs() >= correlation_threshold].index

# Añadir la variable objetivo a las variables a mantener
variables_to_keep = variables_to_keep.insert(0, target_var)

# Crear un nuevo DataFrame con solo las variables seleccionadas
filtered_df = df[variables_to_keep]

print("Variables mantenidas:", variables_to_keep)
df= filtered_df
print(df)

# %% [markdown]
# ### Segmentando los datos fuente en un conjunto de entrenamiento, uno de validación y uno de prueba.
# 
# Considerare un 60% de los datos para entrenamiento, un 20% para validación y el restante 20% para prueba.Para ello se hará uso de la función train_test_split de la libreria sklearn. Esto se realizará dos veces para obtener los 3 conjuntos que necesitamos.
# 

# %%
X = df.drop('exited', axis=1)
y = df['exited']


# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=12345)
X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=12345)

# %% [markdown]
# Antes de proceder con el entrenamiento de los modelos, se considera efectuar una estandarizacion de las variables numericas.

# %%
# Instanciar el StandardScaler
scaler = StandardScaler()

# Ajustar y transformar el conjunto de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)

# Solo transformar el conjunto de prueba (NO ajustar nuevamente)
X_test_scaled = scaler.transform(X_test)

# %%
# Persistimos los cambios para trabajar en el entrenamiento
X_train = X_train_scaled
X_test = X_test_scaled


# %%
# Crear un pipeline para el manejo de los modelos
pipelines = {
    'rf': Pipeline([
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ]),
    'lgbm': Pipeline([
        ('classifier', LGBMClassifier(random_state=42))
    ]),
    'xgb': Pipeline([
        ('classifier', XGBClassifier(random_state=42))
    ]),
    'catboost': Pipeline([
        ('classifier', CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced'))
    ])
}


# %%
param_grids = {
    'rf': {'classifier__n_estimators': [10,50, 150],
           'classifier__max_depth': [3, 4, 5],
           'classifier__min_samples_leaf': [1, 2, 4],
           'classifier__min_samples_split': [2, 5, 10]},
    'lgbm': {'classifier__n_estimators': [10,50, 100],
              'classifier__max_depth': [3, 5, 10],
           'classifier__num_leaves': [10, 30, 50],
           'classifier__learning_rate': [0.1, 0.05, 0.01]},
    'xgb': {'classifier__n_estimators': [10, 50, 150],
           'classifier__subsample': [0.8, 0.9, 1],
           'classifier__gamma': [0, 0.1, 0.200],
           'classifier__learning_rate': [0.1, 0.01, 0.001],
           'classifier__colsample_bytree': [0.8, 0.9, 1],
           'classifier__max_depth': [3, 5, 10]},
    'catboost': {'classifier__iterations': [10, 50, 150],
                 'classifier__depth': [4, 6, 10],
                 'classifier__l2_leaf_reg': [1, 3, 5, 7, 9],
                 }
}

# %%
# Inicializar una lista para almacenar los resultados
results = []

stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, pipeline in pipelines.items():
    grid_search = GridSearchCV(pipeline, param_grids[model_name], cv=stratified_k_fold,
                               scoring='roc_auc', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    # Evaluar el modelo en el conjunto de prueba
    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
    y_pred = grid_search.predict(X_test)
    
        # Aplicar umbral a las probabilidades
    umbral = 0.5  # Umbral de 0.5 para clasificación binaria
    y_pred_binario = np.where(y_pred_proba >= umbral, 1, 0)

    # Calcular la precisión con las predicciones binarias
    acc_score = accuracy_score(y_test, y_pred_binario)
    f1 = f1_score(y_test, y_pred_binario)
    precision = precision_score(y_test, y_pred_binario)
    recall = recall_score(y_test, y_pred_binario)
    roc_auc = roc_auc_score(y_test, y_pred_binario)
    
    # Guardar los resultados en la lista
    results.append({
        'Modelo': model_name,
        'AUC-ROC': roc_auc,
        'Exactitud': acc_score,
        'f1': f1,
        'Precision': precision,
        'recall': recall,
        'Mejores Parámetros': grid_search.best_params_,
        'Mejor AUC-ROC': grid_search.best_score_
    })

# Crear el DataFrame de resultados
results_df = pd.DataFrame(results)

# Mostrar la tabla de resultados
print(results_df)


# %%
print(results_df)

# %%
best_model = pipelines[results_df.sort_values('AUC-ROC', ascending=False).iloc[0]['Modelo']]
best_model.fit(X_train, y_train)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = best_model.predict(X_test)

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()


