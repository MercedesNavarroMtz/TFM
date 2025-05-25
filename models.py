#%%
from utils import *
from functools import reduce
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#%%
# CARGA
sensor_nu1 = obtener_datos('https://www.ctndatabase.ctnaval.com/aquamore/load_sensors/?sensor_name=nu1')
sensor_nu2 = obtener_datos('https://www.ctndatabase.ctnaval.com/aquamore/load_sensors/?sensor_name=nu2')
sensor_nu3 = obtener_datos('https://www.ctndatabase.ctnaval.com/aquamore/load_sensors/?sensor_name=nu3')
sensor_nu4 = obtener_datos('https://www.ctndatabase.ctnaval.com/aquamore/load_sensors/?sensor_name=nu4')

# Visualizamos los datos y sus estadísticas básicas
bassic_preprocessing(sensor_nu1)
bassic_preprocessing(sensor_nu2)
bassic_preprocessing(sensor_nu3)
bassic_preprocessing(sensor_nu4)

# Eliminación de duplicados
sensor_nu1 = sensor_nu1.drop_duplicates()
sensor_nu2 = sensor_nu2.drop_duplicates()
sensor_nu3 = sensor_nu3.drop_duplicates()
sensor_nu4 = sensor_nu4.drop_duplicates()

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



columnas = [col for col in df_carga_merged.columns if col.startswith('weight') or col == 'date_time'] # Dejamos solo las columnas weight y date_time
df_carga_merged = df_carga_merged.loc[:, columnas]
df_carga_merged = df_carga_merged.rename(columns={'weight': 'weight_nu3'}) # La weight_nu3 no se habia puesto

representacion_umbral_carga(df_carga_merged, 'weight_nu4')

df_carga_merged = df_carga_merged[df_carga_merged['date_time'].dt.year == 2024] #  DEJAMOS SOLO VALORES DE 2024 porque los de 2023 son muy pcoos valores y outliers
representacion_umbral_carga(df_carga_merged, 'weight_nu4')


#Guardamos los datos en csv
# df_carga_merged.to_csv('carga.csv', index=False) 

#%% RED


sensor_43 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=43")
sensor_44 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=44")
sensor_45 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=45")
sensor_46 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=46")
sensor_47 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=47")
sensor_48 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=48")
sensor_49 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=49")

bassic_preprocessing(sensor_43)
bassic_preprocessing(sensor_44)
bassic_preprocessing(sensor_45)
bassic_preprocessing(sensor_46)
bassic_preprocessing(sensor_47)
bassic_preprocessing(sensor_48)
bassic_preprocessing(sensor_49)

sensor_43 = sensor_43.drop_duplicates()
sensor_44 = sensor_44.drop_duplicates()
sensor_45 = sensor_45.drop_duplicates()
sensor_46 = sensor_46.drop_duplicates()
sensor_47 = sensor_47.drop_duplicates()
sensor_48 = sensor_48.drop_duplicates()
sensor_49 = sensor_49.drop_duplicates()

#%%
def representar_sensores_red(inicio, fin, aplicar_limpieza=False):
    """
    Procesa y grafica los datos de pitch y roll para sensores en el rango dado.

    Args:
        inicio (int): Número inicial del sensor
        fin (int): Número final del sensor
        aplicar_limpieza (bool): Si True, elimina outliers de mayo 2024.
    """
    for i in range(inicio, fin + 1):
        df = globals()[f'sensor_{i}'].copy()
        df['date_time'] = pd.to_datetime(df['date_time'])

        if aplicar_limpieza:
            df = remove_outliers_may_2024(df, 'pitch')
            df = remove_outliers_may_2024(df, 'roll')

        mean_pitch = df['pitch'].mean()
        mean_roll = df['roll'].mean()

        df['pitch_color'] = df['pitch'].apply(lambda x: color_points(x, mean_pitch))
        df['roll_color'] = df['roll'].apply(lambda x: color_points(x, mean_roll))

        # Gráfico
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
        fig.suptitle(f'Sensor {i}', fontsize=18)

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
# Función que representa los sensores coloreando los puntos
representar_sensores_red(43, 49)

#Aplicamos un parametro para limpiar los outliers de 2024 a la funcion 
representar_sensores_red(43, 49, aplicar_limpieza=True)

#Como tenemos que mergear y todos los datos tienen los mismos parametros renombramos con su n  de sensor

sensor_43 = renombrar_columnas(sensor_43, 43)
sensor_44 = renombrar_columnas(sensor_44, 44)
sensor_45 = renombrar_columnas(sensor_45, 45)
sensor_46 = renombrar_columnas(sensor_46, 46)
sensor_47 = renombrar_columnas(sensor_47, 47)
sensor_48 = renombrar_columnas(sensor_48, 48)
sensor_49 = renombrar_columnas(sensor_49, 49)

dfs = [sensor_43, sensor_44, sensor_45, sensor_46, sensor_47, sensor_48, sensor_49]
df_red_merged = reduce(lambda left, right: pd.merge(left, right, on='date_time', how='inner'), dfs) # Unimos solo los comunes


#Guardamos en CSV
# df_red_merged.to_csv('red.csv', index=False)

#%% FLOTABILIDAD

sensor_60 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=60")
sensor_61 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=61")
sensor_62 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=62")
sensor_63 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=63")

bassic_preprocessing(sensor_60)
bassic_preprocessing(sensor_61)
bassic_preprocessing(sensor_62)
bassic_preprocessing(sensor_63)

sensor_60 = sensor_60.drop_duplicates()
sensor_61 = sensor_61.drop_duplicates()
sensor_62 = sensor_62.drop_duplicates()
sensor_63 = sensor_63.drop_duplicates()

# Función que representa los sensores coloreando los puntos
representar_sensores_red(60, 63)

#Aplicamos un parametro para limpiar los outliers de 2024 a la funcion 
representar_sensores_red(60, 63, aplicar_limpieza=True)

# Renombrar columnas para el merge
sensor_60 = renombrar_columnas(sensor_60, 60)
sensor_61 = renombrar_columnas(sensor_61, 61)
sensor_62 = renombrar_columnas(sensor_62, 62)
sensor_63 = renombrar_columnas(sensor_63, 63)

dfs = [sensor_60, sensor_61, sensor_62, sensor_63]
df_flot_merged = reduce(lambda left, right: pd.merge(left, right, on='date_time', how='inner'), dfs) #Con el inner unimos solo los comunes

# df_flot_merged.to_csv('flot.csv', index=False)

#%% CORRIENTE

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
min_speed = min([speed.min() for speed in all_speeds]) # Para poner limite al gráfico (todas y =)
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

corr_aqua101 = corr_aqua101.iloc[:, :-18] # Eliminamos variables que no sirven (de la 12 en adelante)

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

#Guardamos en CSV
# corr_aqua101.to_csv('corriente.csv', index=False)

# %% MODELOSSSS------------------------::

# %% METODO INTERPOLADO

# merges sucesivos
df_merged = df_carga_merged.merge(df_red_merged, on='date_time', how='outer') \
               .merge(corr_aqua101, on='date_time', how='outer') \
               .merge(df_flot_merged, on='date_time', how='outer')

# Revisamos y eliminamos las columnas id y sensor que no aportan info
df_merged = df_merged.loc[:, ~df_merged.columns.get_level_values(0).str.startswith(('id_', 'sensor_'))]
df_merged[df_merged.duplicated()]


df_merged = df_merged.set_index('date_time')
df_merged = df_merged.interpolate(method='time', order=2)
df_merged = df_merged.reset_index()  #
df_merged = df_merged.dropna()

# Escalado de datos
variables = df_merged.select_dtypes(include='number').columns
scaler = StandardScaler()
df_scaled_interpolado = df_merged.copy()
df_scaled_interpolado[variables] = scaler.fit_transform(df_merged[variables])

# df_scaled_interpolado.to_csv('mergeados_interpolados_normalizados.csv', index=False)

# %% METODO MERGE FECHAS CERCANAS

# merge sucesivo
merged_fecha = pd.merge_asof(df_carga_merged, df_red_merged, on='date_time', direction='nearest')
merged_fecha = pd.merge_asof(merged_fecha, df_flot_merged, on='date_time', direction='nearest')
merged_fecha = pd.merge_asof(merged_fecha, corr_aqua101, on='date_time', direction='nearest')

merged_fecha = merged_fecha.dropna()

# Escalado de datos
variables = merged_fecha.select_dtypes(include='number').columns
scaler = StandardScaler()
df_scaled_fecha = merged_fecha.copy()
df_scaled_fecha[variables] = scaler.fit_transform(merged_fecha[variables])

# df_scaled_fecha.to_csv('mergeados_normalizados.csv', index=False)

# %% GRID SEARCH - entrenamiento

parametros_rf = {
    'n_estimators': [100, 150, 200],  # Número de árboles 
    'max_depth': [5, 10, None],        # Profundidad de los árboles
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

modelos = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'LinearRegression': LinearRegression()
}


def probar_modelos_con_gridsearch(df, columna_objetivo, columnas_features, modelos, parametros, test_size=0.3):
    X = df[columnas_features]
    y = df[columna_objetivo]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)

    resultados = {}

    for nombre, modelo in modelos.items():
        print(f"Optimizando {nombre}...")

        if nombre in parametros and parametros[nombre]: 
            grid_search = GridSearchCV(estimator=modelo, param_grid=parametros[nombre], cv=5, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            mejor_modelo = grid_search.best_estimator_
            mejores_params = grid_search.best_params_
        else:
            # Para modelos sin parámetros (LinearRegression)
            mejor_modelo = modelo
            mejor_modelo.fit(X_train, y_train)
            mejores_params = {}

        y_train_pred = mejor_modelo.predict(X_train)
        y_test_pred = mejor_modelo.predict(X_test)

        # entrenamiento
        mse_train = mean_squared_error(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        # test
        mse_test = mean_squared_error(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)

        resultados[nombre] = {
            'Mejores Parámetros': mejores_params,
            'MSE Train': mse_train,
            'MAE Train': mae_train,
            'R2 Train': r2_train,
            'MSE Test': mse_test,
            'MAE Test': mae_test,
            'R2 Test': r2_test,

        }

        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.7)
        plt.xlabel('Valores reales', fontsize=12)
        plt.ylabel('Predicciones', fontsize=12)
        plt.title(f'Valores reales vs predichos ({nombre})', fontsize=14)
        plt.grid(True, alpha=0.3)

        min_val = min(min(y_test), min(y_test_pred))
        max_val = max(max(y_test), max(y_test_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        plt.tight_layout()
        plt.show()


    return resultados

#%%
interpolados = pd.read_csv('mergeados_interpolados_normalizados.csv')
por_fecha = pd.read_csv('mergeados_normalizados.csv')
#%%
columna_objetivo = 'weight_nu4'

columnas_features = por_fecha.drop(columns=['weight_nu1', 'weight_nu2','weight_nu3','weight_nu4','date_time']).columns

resultados_nu = probar_modelos_con_gridsearch(por_fecha,columna_objetivo,columnas_features,modelos,
    {
        'RandomForest': parametros_rf,
        'GradientBoosting': parametros_gb,
        'LinearRegression': {}
    }
)

df_resultados = pd.DataFrame(resultados_nu).T 
nombre_df = 'por_fecha'
nombre_archivo = f"resultados_{columna_objetivo}_{nombre_df}.xlsx"
df_resultados.to_excel(nombre_archivo)





# %% CROSS VALIDATION

# Configuramod el modelo GradientBoostingRegressor con los mejores parámetros para cada tecnica
mejor_gb_inter = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)

mejor_gb_merge = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.1,
    subsample=1.0,
    random_state=42
)




X_mergeados = por_fecha.drop(columns=['weight_nu1', 'weight_nu2', 'weight_nu3', 'weight_nu4', 'date_time'])
y_mergeados = por_fecha['weight_nu2']

X_interpolados = interpolados.drop(columns=['weight_nu1', 'weight_nu2','weight_nu3','weight_nu4','date_time'])
y_interpolados = interpolados['weight_nu2']

resultados_interpolados = evaluar_con_validacion_cruzada(X_interpolados, y_interpolados, mejor_gb_inter)
resultados_mergeados = evaluar_con_validacion_cruzada(X_mergeados, y_mergeados, mejor_gb_merge)


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

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(range(1, 6), resultados_interpolados['Resultados por fold']['R2'], color='blue', alpha=0.7)
plt.ylim(0.5, 1.0) 
plt.title('R² por fold - Interpolados')
plt.xlabel('Fold')
plt.ylabel('R²')

plt.subplot(1, 2, 2)
plt.bar(range(1, 6), resultados_mergeados['Resultados por fold']['R2'], color='green', alpha=0.7)
plt.ylim(0.5, 1.0) 
plt.title('R² por fold - Mergeados')
plt.xlabel('Fold')
plt.ylabel('R²')

plt.tight_layout()
plt.show()

# %% Análisis de los residuos
X_mergeados = por_fecha.drop(columns=['weight_nu1', 'weight_nu2', 'weight_nu3', 'weight_nu4', 'date_time'])
y_mergeados = por_fecha['weight_nu4']

X_interpolados = interpolados.drop(columns=['weight_nu1', 'weight_nu2','weight_nu3','weight_nu4','date_time'])
y_interpolados = interpolados['weight_nu4']

# modelo con datos interpolados
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X_interpolados, y_interpolados, test_size=0.3, random_state=42)
modelo_interpolado = GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=7, learning_rate=0.1, subsample=0.9)
modelo_interpolado.fit(X_train_i, y_train_i)
y_pred_i = modelo_interpolado.predict(X_test_i)

#  modelo con datos mergeados
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_mergeados, y_mergeados, test_size=0.3, random_state=42)
modelo_mergeado = GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=7, learning_rate=0.05, subsample=0.8)
modelo_mergeado.fit(X_train_m, y_train_m)
y_pred_m = modelo_mergeado.predict(X_test_m)

resultados_interpolados = analizar_residuos(y_test_i, y_pred_i, titulo="Modelo datos interpolados")
resultados_mergeados = analizar_residuos(y_test_m, y_pred_m, titulo="Modelo datos mergeados")

comparacion = comparar_residuos_modelos(
    [(y_test_i, y_pred_i), (y_test_m, y_pred_m)],
    nombres_modelos=["Interpolados", "Mergeados"]
)
print(comparacion)

# %%
