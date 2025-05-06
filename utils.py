
#%% FUNCIONES CARGA
# PETICIÓN GET PARA OBTENER LOS DATOS ===================================================================
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def obtener_datos(url):

    all_results = []

    while url:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            all_results.extend(results)
            url = data.get("next")  # Avanza a la siguiente página si existe
        else:
            print(f"Error: {response.status_code}")
            break

    df = pd.DataFrame(all_results)

    df['date_time'] = pd.to_datetime(df['date_time'])
    return df

def bassic_preprocessing(df):
    '''Función que hace un resumen general de un DataFrame con sensores'''
    
    print(">> Información general del DataFrame:")
    print(df.info())
    print("\n>> Estadísticas descriptivas:")
    print(df.describe())
    
    print("\n>> Valores nulos por columna:")
    print(df.isnull().sum())

    print(f"\n>> Filas duplicadas: {df.duplicated().sum()}")

    print("\n>> Primeras 5 filas del DataFrame:")
    print(df.head())


def representacion_umbral_carga(df, sensor_name):
    x = df['date_time']
    y = df[sensor_name]
    mask = y > 85000

    plt.figure(figsize=(14, 6))
    plt.scatter(x[~mask], y[~mask], alpha=0.6)
    plt.scatter(x[mask], y[mask], color='red', label='> 85000', alpha=0.8)
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

    plt.xlabel('Fecha',fontsize=18)
    plt.ylabel('Carga [kg]',fontsize=18)
    plt.title(f'Serie temporal de sensor carga {sensor_name}', fontsize=15)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.ylim()
    plt.show()


# %% OBTENEMOS DATOS
sensor_nu1 = obtener_datos('https://www.ctndatabase.ctnaval.com/aquamore/load_sensors/?sensor_name=nu1')
sensor_nu2 = obtener_datos('https://www.ctndatabase.ctnaval.com/aquamore/load_sensors/?sensor_name=nu2')
sensor_nu3 = obtener_datos('https://www.ctndatabase.ctnaval.com/aquamore/load_sensors/?sensor_name=nu3')
sensor_nu4 = obtener_datos('https://www.ctndatabase.ctnaval.com/aquamore/load_sensors/?sensor_name=nu4')

# Visualizamos los datos y sus estadísticas básicas
bassic_preprocessing(sensor_nu1)
bassic_preprocessing(sensor_nu2)
bassic_preprocessing(sensor_nu3)
bassic_preprocessing(sensor_nu4)

#%%
sensor_nu1 = sensor_nu1.drop_duplicates()
sensor_nu2 = sensor_nu2.drop_duplicates()
sensor_nu3 = sensor_nu3.drop_duplicates()
sensor_nu4 = sensor_nu4.drop_duplicates()



#%%
# Filtramos a las fechas comunes
start_date = pd.to_datetime("2023-07-11")
end_date = pd.to_datetime("2024-05-02")

sensor_nu1 = sensor_nu1[(sensor_nu1['date_time'] >= start_date) & (sensor_nu1['date_time'] <= end_date)]
sensor_nu2 = sensor_nu2[(sensor_nu2['date_time'] >= start_date) & (sensor_nu2['date_time'] <= end_date)]
sensor_nu3 = sensor_nu3[(sensor_nu3['date_time'] >= start_date) & (sensor_nu3['date_time'] <= end_date)]
sensor_nu4 = sensor_nu4[(sensor_nu4['date_time'] >= start_date) & (sensor_nu4['date_time'] <= end_date)]




# Hemos dejado solo las fechas comines y hacemos un merge para quedarnos con un solo df
df_carga_merged = sensor_nu1.merge(sensor_nu2, on='date_time', suffixes=('_nu1', '_nu2')) \
                     .merge(sensor_nu3, on='date_time', suffixes=('', '_nu3')) \
                     .merge(sensor_nu4, on='date_time', suffixes=('', '_nu4'))




columnas = [col for col in df_carga_merged.columns if col.startswith('weight') or col == 'date_time']
df_carga_merged = df_carga_merged.loc[:, columnas]

df_carga_merged = df_carga_merged.rename(columns={'weight': 'weight_nu3'})
#%%
representacion_umbral_carga(df_carga_merged, 'weight_nu4')
#%%
#  DEJAMOS SOLO VALORES DE 2024 porque los de 2023 son muy pcoos valores y outliers
df_carga_merged = df_carga_merged[df_carga_merged['date_time'].dt.year == 2024]
#%%
representacion_umbral_carga(df_carga_merged, 'weight_nu4')

#%%
df_carga_merged.to_csv('carga.csv', index=False) 



# %%CORRENTÍMETRO ===================================================================

corr_aqua101=obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/corr_sensor")
bassic_preprocessing(corr_aqua101)


del corr_aqua101['temperature']
del corr_aqua101['id']

# Grafico con todas las corrientes --------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

for i in range(1, 12):
    speed = corr_aqua101[f'speed{i}']

    ax.plot(corr_aqua101['date_time'], speed, label=f'Speed {i}')

ax.set_xlabel('Fecha')
ax.set_ylabel('Velocidad de la corriente (m/s)')
plt.xticks(rotation=45)
plt.grid(True)
ax.legend()
plt.show()

# Grafico con las corrientes por separado ----------------------------

all_speeds = [corr_aqua101[f'speed{i}'] for i in range(1, 12)]
min_speed = min([speed.min() for speed in all_speeds])
max_speed = max([speed.max() for speed in all_speeds])

fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(12, 18))  # 6x2 
axes = axes.flatten()

for i in range(1, 12):
    speed = corr_aqua101[f'speed{i}']

    ax = axes[i - 1]

    ax.plot(corr_aqua101['date_time'], speed, label=f'Speed {i}', color='b')

    ax.set_title(f'Velocidad de la corriente - Speed{i}')
    ax.set_xlabel('Fecha')
    ax.set_ylabel('Velocidad (m/s)')
    ax.set_ylim(min_speed, max_speed)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True)

axes[11].axis('off')
plt.tight_layout()
plt.show()

corr_aqua101 = corr_aqua101.iloc[:, :-18] # Eliminamos variables que no sirven

#Graficos de dirección - rosa de los vientos --------------------

fig, axes = plt.subplots(nrows=6, ncols=2, figsize=(12, 18))  # 6x2 

axes = axes.flatten()

for i in range(1, 12):
    direction = corr_aqua101[f'direction{i}']
    ax = axes[i - 1]
    direction_rad = np.deg2rad(direction)
    ax = plt.subplot(6, 2, i, projection='polar')
    ax.hist(direction_rad, bins=8, edgecolor='black', color='blue', alpha=0.7)
    ax.set_title(f'Dirección de la corriente - Direction{i}')

axes[11].axis('off')

plt.tight_layout()

plt.show()


# corr_aqua101.to_csv('corriente.csv', index=False)

#%% RED 


from functools import reduce
sensor_43 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=43")
sensor_44 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=44")
sensor_45 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=45")
sensor_46 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=46")
sensor_47 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=47")
sensor_48 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=48")
sensor_49 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=49")
# %%
sensor_43 = sensor_43.drop_duplicates()
sensor_44 = sensor_44.drop_duplicates()
sensor_45 = sensor_45.drop_duplicates()
sensor_46 = sensor_46.drop_duplicates()
sensor_47 = sensor_47.drop_duplicates()
sensor_48 = sensor_48.drop_duplicates()
sensor_49 = sensor_49.drop_duplicates()


#%%
bassic_preprocessing(sensor_43)
bassic_preprocessing(sensor_44)
bassic_preprocessing(sensor_45)
bassic_preprocessing(sensor_46)
bassic_preprocessing(sensor_47)
bassic_preprocessing(sensor_48)
bassic_preprocessing(sensor_49)



# %% REPRESENTACIONES DE PITCH Y ROLL


# Función para colorear los puntos
def color_points(value, mean):
    if value > mean + 10:
        return 'green'
    elif value < mean - 10:
        return 'red'
    else:
        return 'black'
    
def remove_outliers_may_2024(df, column):
    mask_may = (df['date_time'].dt.year == 2024) & (df['date_time'].dt.month == 5)
    mean_val = df.loc[mask_may, column].mean()
    std_val = df.loc[mask_may, column].std()

    upper_limit = mean_val + 10
    lower_limit = mean_val - 10

    df = df[~((mask_may) & ((df[column] > upper_limit) | (df[column] < lower_limit)))]

    return df

#%%

# Bucle sobre los sensores
for i in range(43, 50):
    df = globals()[f'sensor_{i}']  # Accede al DataFrame llamado sensor_i
    df['date_time'] = pd.to_datetime(df['date_time'])

    mean_pitch = df['pitch'].mean()
    mean_roll = df['roll'].mean()

    df['pitch_color'] = df['pitch'].apply(lambda x: color_points(x, mean_pitch))
    df['roll_color'] = df['roll'].apply(lambda x: color_points(x, mean_roll))

    # Crear la figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(f'Sensor {i}', fontsize=16)

    ax1.scatter(df['date_time'], df['pitch'], c=df['pitch_color'])
    ax1.axhline(mean_pitch, color='gray', linestyle='--', label='Media Pitch')
    ax1.set_ylabel('Pitch')
    ax1.legend()
    ax1.set_title('Pitch')

    ax2.scatter(df['date_time'], df['roll'], c=df['roll_color'])
    ax2.axhline(mean_roll, color='gray', linestyle='--', label='Media Roll')
    ax2.set_ylabel('Roll')
    ax2.set_xlabel('Fecha')
    ax2.legend()
    ax2.set_title('Roll')

    plt.tight_layout()
    plt.show()

# %%

for i in range(43, 50):
    df = globals()[f'sensor_{i}'].copy()
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Eliminar outliers de mayo 2024
    df = remove_outliers_may_2024(df, 'pitch')
    df = remove_outliers_may_2024(df, 'roll')

    # Calcular medias para todo el conjunto (tras limpieza)
    mean_pitch = df['pitch'].mean()
    mean_roll = df['roll'].mean()

    # Colorear puntos
    df['pitch_color'] = df['pitch'].apply(lambda x: color_points(x, mean_pitch))
    df['roll_color'] = df['roll'].apply(lambda x: color_points(x, mean_roll))

    # Crear figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(f'Sensor {i}', fontsize=16)

    ax1.scatter(df['date_time'], df['pitch'], c=df['pitch_color'])
    ax1.axhline(mean_pitch, color='gray', linestyle='--', label='Media Pitch')
    ax1.set_ylabel('Pitch')
    ax1.legend()
    ax1.set_title('Pitch')

    ax2.scatter(df['date_time'], df['roll'], c=df['roll_color'])
    ax2.axhline(mean_roll, color='gray', linestyle='--', label='Media Roll')
    ax2.set_ylabel('Roll')
    ax2.set_xlabel('Fecha')
    ax2.legend()
    ax2.set_title('Roll')

    plt.tight_layout()
    plt.show()


#%% Mergeamos

def renombrar_columnas(df, sensor_id):
    df = df.copy()
    df.columns = [
        col if col == "date_time" else f"{col}_{sensor_id}"
        for col in df.columns
    ]
    return df
sensor_43 = renombrar_columnas(sensor_43, 43)
sensor_44 = renombrar_columnas(sensor_44, 44)
sensor_45 = renombrar_columnas(sensor_45, 45)
sensor_46 = renombrar_columnas(sensor_46, 46)
sensor_47 = renombrar_columnas(sensor_47, 47)
sensor_48 = renombrar_columnas(sensor_48, 48)
sensor_49 = renombrar_columnas(sensor_49, 49)
#%%
from functools import reduce

dfs = [sensor_43, sensor_44, sensor_45, sensor_46, sensor_47, sensor_48, sensor_49]
df_red_merged = reduce(lambda left, right: pd.merge(left, right, on='date_time', how='inner'), dfs)

#%%
df_red_merged.to_csv('red.csv', index=False)

# %%




# FOLTABILIDAD
sensor_60 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=60")
sensor_61 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=61")
sensor_62 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=62")
sensor_63 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=63")

#%%
sensor_60 = sensor_60.drop_duplicates()
sensor_61 = sensor_61.drop_duplicates()
sensor_62 = sensor_62.drop_duplicates()
sensor_63 = sensor_63.drop_duplicates()

bassic_preprocessing(sensor_60)
bassic_preprocessing(sensor_61)
bassic_preprocessing(sensor_62)
bassic_preprocessing(sensor_63)

#%%
# Bucle sobre los sensores
for i in range(60, 64):
    df = globals()[f'sensor_{i}'] 
    df['date_time'] = pd.to_datetime(df['date_time'])

    mean_pitch = df['pitch'].mean()
    mean_roll = df['roll'].mean()

    df['pitch_color'] = df['pitch'].apply(lambda x: color_points(x, mean_pitch))
    df['roll_color'] = df['roll'].apply(lambda x: color_points(x, mean_roll))

    # Crear la figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(f'Sensor {i}', fontsize=16)

    ax1.scatter(df['date_time'], df['pitch'], c=df['pitch_color'])
    ax1.axhline(mean_pitch, color='gray', linestyle='--', label='Media Pitch')
    ax1.set_ylabel('Pitch')
    ax1.legend()
    ax1.set_title('Pitch')

    ax2.scatter(df['date_time'], df['roll'], c=df['roll_color'])
    ax2.axhline(mean_roll, color='gray', linestyle='--', label='Media Roll')
    ax2.set_ylabel('Roll')
    ax2.set_xlabel('Fecha')
    ax2.legend()
    ax2.set_title('Roll')

    plt.tight_layout()
    plt.show()
# %%
for i in range(60, 64):
    df = globals()[f'sensor_{i}'].copy()
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Eliminar outliers de mayo 2024
    df = remove_outliers_may_2024(df, 'pitch')
    df = remove_outliers_may_2024(df, 'roll')

    # Calcular medias para todo el conjunto (tras limpieza)
    mean_pitch = df['pitch'].mean()
    mean_roll = df['roll'].mean()

    # Colorear puntos
    df['pitch_color'] = df['pitch'].apply(lambda x: color_points(x, mean_pitch))
    df['roll_color'] = df['roll'].apply(lambda x: color_points(x, mean_roll))

    # Crear figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.suptitle(f'Sensor {i}', fontsize=16)

    ax1.scatter(df['date_time'], df['pitch'], c=df['pitch_color'])
    ax1.axhline(mean_pitch, color='gray', linestyle='--', label='Media Pitch')
    ax1.set_ylabel('Pitch')
    ax1.legend()
    ax1.set_title('Pitch')

    ax2.scatter(df['date_time'], df['roll'], c=df['roll_color'])
    ax2.axhline(mean_roll, color='gray', linestyle='--', label='Media Roll')
    ax2.set_ylabel('Roll')
    ax2.set_xlabel('Fecha')
    ax2.legend()
    ax2.set_title('Roll')

    plt.tight_layout()
    plt.show()


# %%
sensor_60 = renombrar_columnas(sensor_60, 60)
sensor_61 = renombrar_columnas(sensor_61, 61)
sensor_62 = renombrar_columnas(sensor_62, 62)
sensor_63 = renombrar_columnas(sensor_63, 63)

dfs = [sensor_60, sensor_61, sensor_62, sensor_63]
df_flot_merged = reduce(lambda left, right: pd.merge(left, right, on='date_time', how='inner'), dfs)

df_flot_merged.to_csv('flot.csv', index=False)
# %% MODELOSSSS------------------------::

# %% METODO INTERPOLADO
# Realizar merges sucesivos
df_merged = df_carga_merged.merge(df_red_merged, on='date_time', how='outer') \
               .merge(corr_aqua101, on='date_time', how='outer') \
               .merge(df_flot_merged, on='date_time', how='outer')

#%%

# Revisamos y eliminamos las columnas id y sensor que no aportan info
df_merged = df_merged.loc[:, ~df_merged.columns.get_level_values(0).str.startswith(('id_', 'sensor_'))]
df_merged[df_merged.duplicated()]


#Interpolamos para quitar valores vacíos
df_merged = df_merged.set_index('date_time')
df_merged = df_merged.interpolate(method='time', order=2)
df_merged = df_merged.reset_index()  #


df_merged = df_merged.dropna()

# Escalado de datos
from sklearn.preprocessing import StandardScaler

variables = df_merged.select_dtypes(include='number').columns

scaler = StandardScaler()

df_scaled_interpolado = df_merged.copy()
df_scaled_interpolado[variables] = scaler.fit_transform(df_merged[variables])


# df_scaled_interpolado.to_csv('mergeados_interpolados_normalizados.csv', index=False)

# %% METODO MERGE FECHAS CERCANAS

# Realizar el merge sucesivo entre los DataFrames
merged_fecha = pd.merge_asof(df_carga_merged, df_red_merged, on='date_time', direction='nearest')
merged_fecha = pd.merge_asof(merged_fecha, df_flot_merged, on='date_time', direction='nearest')
merged_fecha = pd.merge_asof(merged_fecha, corr_aqua101, on='date_time', direction='nearest')

merged_fecha = merged_fecha.dropna()

# %%

# Seleccionar solo las columnas numéricas (excluyendo date_time)
variables = merged_fecha.select_dtypes(include='number').columns

# Crear el escalador
scaler = StandardScaler()

# Aplicar el escalado
df_scaled_fecha = merged_fecha.copy()
df_scaled_fecha[variables] = scaler.fit_transform(merged_fecha[variables])

#%%
# df_scaled_fecha.to_csv('mergeados_normalizados.csv', index=False)

# %% GRID SEARCH 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Definir los parámetros para Grid Search
parametros_rf = {
    'n_estimators': [100, 150, 200],  # Número de árboles en el bosque
    'max_depth': [5, 10, None],        # Profundidad máxima de los árboles
    'min_samples_split': [2, 5, 10],   # Mínimo número de muestras requeridas para dividir un nodo
    'min_samples_leaf': [1, 2, 4],     # Mínimo número de muestras requeridas en una hoja
    'bootstrap': [True, False]         # Si usar o no muestreo bootstrap
}

parametros_gb = {
    'n_estimators': [100, 150, 200],  # Número de árboles en el modelo
    'learning_rate': [0.01, 0.05, 0.1], # Tasa de aprendizaje
    'max_depth': [3, 5, 7],            # Profundidad máxima de los árboles
    'subsample': [0.8, 0.9, 1.0]       # Proporción de muestras usadas para entrenar cada árbol
}

# Definir los modelos a utilizar
modelos = {
    # 'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
    # 'LinearRegression': LinearRegression()
}

def probar_modelos_con_gridsearch(df, columna_objetivo, columnas_features, modelos, parametros, test_size=0.3):
    X = df[columnas_features]
    y = df[columna_objetivo]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)

    resultados = {}

    for nombre, modelo in modelos.items():
        print(f"Optimizando {nombre}...")

        # Configurar GridSearchCV
        if nombre in parametros and parametros[nombre]:  # Verificar si hay parámetros para el modelo
            grid_search = GridSearchCV(estimator=modelo, param_grid=parametros[nombre], cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            mejor_modelo = grid_search.best_estimator_
            mejores_params = grid_search.best_params_
        else:
            # Para modelos sin parámetros de búsqueda (como LinearRegression)
            mejor_modelo = modelo
            mejor_modelo.fit(X_train, y_train)
            mejores_params = {}

        # Predicciones
        y_train_pred = mejor_modelo.predict(X_train)
        y_test_pred = mejor_modelo.predict(X_test)

        # Métricas en entrenamiento
        mse_train = mean_squared_error(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        # Métricas en test
        mse_test = mean_squared_error(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # Guardar resultados
        resultados[nombre] = {
            'Mejores Parámetros': mejores_params,
            'MSE Train': mse_train,
            'MAE Train': mae_train,
            'R2 Train': r2_train,
            'MSE Test': mse_test,
            'MAE Test': mae_test,
            'R2 Test': r2_test,

        }

        # Gráfico de prueba - ahora usando el nombre correcto del modelo
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.7)

        # Configurar el gráfico con el nombre del modelo actual
        plt.xlabel('Valores Reales', fontsize=12)
        plt.ylabel('Predicciones', fontsize=12)
        plt.title(f'Valores reales vs predichos ({nombre})', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Agregar línea diagonal de referencia
        min_val = min(min(y_test), min(y_test_pred))
        max_val = max(max(y_test), max(y_test_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        plt.tight_layout()
        plt.show()


        #
        # plt.figure(figsize=(14, 6))
        
        # # Serie completa de valores reales
        # plt.plot(por_fecha['date_time'], y, color='blue')

        # # Predicciones solo del conjunto de test
        # plt.plot(y_test.index, y_test_pred, label='Predicciones (Test)', color='orange', linestyle='--')

        # plt.xlabel('Fecha', fontsize=12)
        # plt.ylabel('Valor', fontsize=12)
        # plt.title(f'Serie completa: valor real vs predicción ({nombre})', fontsize=14)
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # plt.tight_layout()
        # plt.show()


    return resultados



#%%
import pandas as pd
interpolados = pd.read_csv('mergeados_interpolados_normalizados.csv')
por_fecha = pd.read_csv('mergeados_normalizados.csv')
#%%

columna_objetivo = 'weight_nu4'

columnas_features = por_fecha.drop(columns=['weight_nu1', 'weight_nu2','weight_nu3','weight_nu4','date_time']).columns

resultados_nu = probar_modelos_con_gridsearch(por_fecha, columna_objetivo, columnas_features, modelos, {
    # 'RandomForest': parametros_rf,
    'GradientBoosting': parametros_gb
    # 'LinearRegression': {}  # Sin parámetros para LinearRegression, ya que no usa GridSearchCV
})

# Mostrar los resultados
df_resultados = pd.DataFrame(resultados_nu).T 

#%%
nombre_df = 'por_fecha'
nombre_archivo = f"resultados_{columna_objetivo}_{nombre_df}.xlsx"
df_resultados.to_excel(nombre_archivo)


# %% CROSS VALIDATION

from sklearn.model_selection import cross_val_score, KFold

def evaluar_con_validacion_cruzada(X, y, modelo, n_folds=5):
    """
    Evalúa un modelo usando validación cruzada y retorna métricas detalladas.
    """
    # Configurar la validación cruzada
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Listas para almacenar resultados
    r2_scores = []
    mse_scores = []
    mae_scores = []
    
    # Para visualizar resultados de cada fold
    fold_results = []
    
    for i, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

       


        # Entrenar modelo
        modelo.fit(X_train_fold, y_train_fold)
        
        # Predecir
        y_val_pred = modelo.predict(X_val_fold)
        
        # Calcular métricas
        r2 = r2_score(y_val_fold, y_val_pred)
        mse = mean_squared_error(y_val_fold, y_val_pred)
        mae = mean_absolute_error(y_val_fold, y_val_pred)
        
        # Guardar resultados
        r2_scores.append(r2)
        mse_scores.append(mse)
        mae_scores.append(mae)
        
        fold_results.append({
            'Fold': i+1,
            'R2': r2,
            'MSE': mse,
            'MAE': mae
        })
    
    # Crear DataFrame con resultados por fold
    df_resultados = pd.DataFrame(fold_results)
    
    # Resultados promedio
    resultados = {
        'R2 medio': sum(r2_scores) / len(r2_scores),
        'MSE medio': sum(mse_scores) / len(mse_scores),
        'MAE medio': sum(mae_scores) / len(mae_scores),
        'R2 desviación': pd.Series(r2_scores).std(),
        'Resultados por fold': df_resultados
    }
    
    return resultados


# Configurar el modelo Random Forest con los mejores parámetros
mejor_gb_inter = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

mejor_gb_merge = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=7,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

# columna_objetivo = 'weight_nu4'

X_mergeados = por_fecha.drop(columns=['weight_nu1', 'weight_nu2', 'weight_nu3', 'weight_nu4', 'date_time'])

y_mergeados = por_fecha['weight_nu1']

X_interpolados = interpolados.drop(columns=['weight_nu1', 'weight_nu2','weight_nu3','weight_nu4','date_time'])
y_interpolados = interpolados['weight_nu1']

# Evaluar ambos conjuntos de datos
print("Evaluando datos interpolados...")
resultados_interpolados = evaluar_con_validacion_cruzada(
    X_interpolados, 
    y_interpolados, 
    mejor_gb_inter
)

print("Evaluando datos mergeados...")
resultados_mergeados = evaluar_con_validacion_cruzada(
    X_mergeados, 
    y_mergeados, 
    mejor_gb_merge
)

#%%
# Comparar resultados
print("\nResultados con datos interpolados:")
print(f"R2 medio: {resultados_interpolados['R2 medio']:.4f} ± {resultados_interpolados['R2 desviación']:.4f}")
print(f"MSE medio: {resultados_interpolados['MSE medio']:.4f}")
print(f"MAE medio: {resultados_interpolados['MAE medio']:.4f}")

print("\nResultados con datos mergeados:")
print(f"R2 medio: {resultados_mergeados['R2 medio']:.4f} ± {resultados_mergeados['R2 desviación']:.4f}")
print(f"MSE medio: {resultados_mergeados['MSE medio']:.4f}")
print(f"MAE medio: {resultados_mergeados['MAE medio']:.4f}")

# Visualizar la variación en los resultados
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(range(1, 6), resultados_interpolados['Resultados por fold']['R2'], color='blue', alpha=0.7)
plt.ylim(0.8, 1.0)  # Ajusta esto según tus valores reales
plt.title('R² por fold - Interpolados')
plt.xlabel('Fold')
plt.ylabel('R²')

plt.subplot(1, 2, 2)
plt.bar(range(1, 6), resultados_mergeados['Resultados por fold']['R2'], color='green', alpha=0.7)
plt.ylim(0.8, 1.0)  # Ajusta esto según tus valores reales
plt.title('R² por fold - Mergeados')
plt.xlabel('Fold')
plt.ylabel('R²')

plt.tight_layout()
plt.show()
# %%


# RESIDUOS

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def analizar_residuos(y_true, y_pred, titulo="Análisis de residuos", mostrar_graficos=True):
    """
    Realiza un análisis completo de residuos para evaluar la calidad de las predicciones.
    
    Args:
        y_true: valores reales
        y_pred: valores predichos
        titulo: título base para los gráficos
        mostrar_graficos: si se deben mostrar los gráficos
    
    Returns:
        dict: diccionario con estadísticas de los residuos
    """
    # Calcular residuos
    residuos = y_true - y_pred
    residuos_abs = np.abs(residuos)
    residuos_norm = residuos / np.std(residuos)
    
    # Estadísticas de los residuos
    stats_residuos = {
        'Media': np.mean(residuos),
        'Mediana': np.median(residuos),
        'Desviación estándar': np.std(residuos),
        'Mínimo': np.min(residuos),
        'Máximo': np.max(residuos),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred)
    }
    
    # Verificar normalidad
    # Test de Shapiro-Wilk
    shapiro_test = stats.shapiro(residuos)
    stats_residuos['Shapiro-Wilk p-value'] = shapiro_test[1]
    stats_residuos['Residuos normales'] = shapiro_test[1] > 0.05
    
    if mostrar_graficos:
        # Crear figura con múltiples subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Residuos vs Valores predichos
        ax1 = fig.add_subplot(221)
        ax1.scatter(y_pred, residuos, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='-')
        ax1.set_xlabel('Valores predichos', fontsize=14)
        ax1.set_ylabel('Residuos', fontsize=14)
        ax1.set_title(f'{titulo}: Residuos vs Predicciones', fontsize=18)
        ax1.grid(True, alpha=0.3)
        
        # Añadir línea de tendencia para ver si hay patrón
        z = np.polyfit(y_pred, residuos, 1)
        p = np.poly1d(z)
        ax1.plot(y_pred, p(y_pred), "r--", alpha=0.8)
        
        # 2. Histograma de residuos
        ax2 = fig.add_subplot(222)
        ax2.hist(residuos, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='-')
        ax2.set_xlabel('Residuos', fontsize=14)
        ax2.set_ylabel('Frecuencia', fontsize=14)
        ax2.set_title(f'{titulo}: Distribución de residuos', fontsize=18)
        
        # 3. QQ Plot para verificar normalidad
        ax3 = fig.add_subplot(223)
        stats.probplot(residuos_norm, dist="norm", plot=ax3)
        ax3.set_title(f'{titulo}: QQ Plot (Normalidad)', fontsize=18)
        ax3.set_xlabel('Theorical quantiles', fontsize=14)
        ax3.set_ylabel('Ordered values', fontsize=14)

        
        # 4. Residuos absolutos vs Valores predichos (para heteroscedasticidad)
        ax4 = fig.add_subplot(224)
        ax4.scatter(y_pred, residuos_abs, alpha=0.6)
        ax4.set_xlabel('Valores predichos', fontsize=14)
        ax4.set_ylabel('|Residuos|', fontsize=14)
        ax4.set_title(f'{titulo}: Residuos Absolutos vs Predicciones', fontsize=18)
        ax4.grid(True, alpha=0.3)
        
        # Línea de tendencia para ver si hay heteroscedasticidad
        z = np.polyfit(y_pred, residuos_abs, 1)
        p = np.poly1d(z)
        ax4.plot(y_pred, p(y_pred), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.show()
    
    return stats_residuos

def comparar_residuos_modelos(modelos_y_datos, nombres_modelos=None):
    """
    Compara los residuos de múltiples modelos.
    
    Args:
        modelos_y_datos: lista de tuplas (y_true, y_pred)
        nombres_modelos: lista de nombres para los modelos
    """
    if nombres_modelos is None:
        nombres_modelos = [f"Modelo {i+1}" for i in range(len(modelos_y_datos))]
    
    # Asegurarse de que tenemos suficientes nombres
    assert len(nombres_modelos) == len(modelos_y_datos), "El número de nombres debe coincidir con el número de modelos"
    
    # Calcular residuos para cada modelo
    todos_residuos = []
    estadisticas = []
    
    for i, (y_true, y_pred) in enumerate(modelos_y_datos):
        residuos = y_true - y_pred
        todos_residuos.append(residuos)
        
        # Calcular estadísticas
        stats_dict = analizar_residuos(y_true, y_pred, titulo=nombres_modelos[i], mostrar_graficos=False)
        stats_dict['Modelo'] = nombres_modelos[i]
        estadisticas.append(stats_dict)
    
    # Crear DataFrame con estadísticas
    df_stats = pd.DataFrame(estadisticas).set_index('Modelo')
    
    # Visualizaciones comparativas
    plt.figure(figsize=(16, 12))
    
    # 1. Boxplot de residuos
    plt.subplot(221)
    plt.boxplot(todos_residuos, labels=nombres_modelos)
    plt.title('Comparación de distribución de residuos', fontsize=18)
    plt.ylabel('Residuos', fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    
    # 2. Violinplot de residuos
    plt.subplot(222)
    plt.violinplot(todos_residuos)
    plt.xticks(range(1, len(nombres_modelos) + 1), nombres_modelos)
    plt.title('Densidad de residuos por modelo', fontsize=18)
    plt.ylabel('Residuos', fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    
    # 3. Histogramas superpuestos
    plt.subplot(223)
    for i, residuos in enumerate(todos_residuos):
        plt.hist(residuos, bins=20, alpha=0.5, label=nombres_modelos[i])
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
    plt.title('Histogramas de residuos superpuestos',fontsize=18)
    plt.xlabel('Residuos', fontsize=14)
    plt.ylabel('Frecuencia', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Gráfico de barras para métricas
    plt.subplot(224)
    metrics = ['RMSE', 'MAE', 'R²']
    x = np.arange(len(metrics))
    width = 0.8 / len(nombres_modelos)
    
    for i, modelo in enumerate(nombres_modelos):
        values = [df_stats.loc[modelo, 'RMSE'], 
                  df_stats.loc[modelo, 'MAE'], 
                  df_stats.loc[modelo, 'R²']]
        plt.bar(x + i*width - width*len(nombres_modelos)/2 + width/2, values, width, label=modelo)
    
    plt.xticks(x, metrics)
    plt.title('Comparación de métricas', fontsize=18)
    plt.ylabel('Valor', fontsize=14)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df_stats

# Ejemplo de uso con tus datos (comentado para que puedas adaptarlo)
"""
# Para el modelo con datos interpolados
resultados_interpolados = analizar_residuos(
    y_test_interpolados, 
    y_pred_interpolados, 
    titulo="Modelo con Datos Interpolados"
)

# Para el modelo con datos mergeados
resultados_mergeados = analizar_residuos(
    y_test_mergeados, 
    y_pred_mergeados, 
    titulo="Modelo con Datos Mergeados"
)

# Comparar ambos modelos
comparacion = comparar_residuos_modelos(
    [(y_test_interpolados, y_pred_interpolados), 
     (y_test_mergeados, y_pred_mergeados)],
    nombres_modelos=["Interpolados", "Mergeados"]
)
"""



#%%

# Después de entrenar y validar los modelos
from sklearn.model_selection import train_test_split

# Para el modelo con datos interpolados
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_interpolados, y_interpolados, test_size=0.3, random_state=42)
modelo_interpolado = GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=5, learning_rate=0.05, subsample=0.8)
modelo_interpolado.fit(X_train_i, y_train_i)
y_pred_i = modelo_interpolado.predict(X_test_i)

# Para el modelo con datos mergeados
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_mergeados, y_mergeados, test_size=0.3, random_state=42)
modelo_mergeado = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=7, learning_rate=0.1,subsample=0.8)
modelo_mergeado.fit(X_train_m, y_train_m)
y_pred_m = modelo_mergeado.predict(X_test_m)

# Analizar residuos
resultados_interpolados = analizar_residuos(y_test_i, y_pred_i, titulo="Modelo datos interpolados")
resultados_mergeados = analizar_residuos(y_test_m, y_pred_m, titulo="Modelo datos mergeados")

# Comparar modelos
comparacion = comparar_residuos_modelos(
    [(y_test_i, y_pred_i), (y_test_m, y_pred_m)],
    nombres_modelos=["Interpolados", "Mergeados"]
)
print(comparacion)

# %%
