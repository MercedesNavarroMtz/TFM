
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
   ''' Función que haga un pequeño resumen para todos los sensores'''
   print('Info del DF', df.info()) 
   print('Estadisticas del DF', df.describe())

def representacion_umbral_carga(df, sensor_name):
    x = df['date_time']
    y = df[sensor_name]
    mask = y > 85000

    plt.figure(figsize=(14, 6))
    plt.scatter(x[~mask], y[~mask], alpha=0.6)
    plt.scatter(x[mask], y[mask], color='red', label='> 85000', alpha=0.8)

    plt.xlabel('Fecha')
    plt.ylabel('Valor')
    plt.title(f'Serie temporal de sensor carga {sensor_name}',fontweight='bold')
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

# Filtramos a las fechas comunes
start_date = pd.to_datetime("2023-07-11")
end_date = pd.to_datetime("2024-05-02")

sensor_nu1 = sensor_nu1[(sensor_nu1['date_time'] >= start_date) & (sensor_nu1['date_time'] <= end_date)]
sensor_nu2 = sensor_nu2[(sensor_nu2['date_time'] >= start_date) & (sensor_nu2['date_time'] <= end_date)]
sensor_nu3 = sensor_nu3[(sensor_nu3['date_time'] >= start_date) & (sensor_nu3['date_time'] <= end_date)]
sensor_nu4 = sensor_nu4[(sensor_nu4['date_time'] >= start_date) & (sensor_nu4['date_time'] <= end_date)]


#  Vemos cuales son las fechas comunes
fechas_comunes = set(sensor_nu1['date_time']).intersection(
    sensor_nu2['date_time'],
    sensor_nu2['date_time'],
    sensor_nu2['date_time']
)

df1_comun = sensor_nu1[sensor_nu1['date_time'].isin(fechas_comunes)]
df2_comun = sensor_nu2[sensor_nu2['date_time'].isin(fechas_comunes)]
df3_comun = sensor_nu3[sensor_nu3['date_time'].isin(fechas_comunes)]
df4_comun = sensor_nu4[sensor_nu4['date_time'].isin(fechas_comunes)]

# Hemos dejado solo las fechas comines y hacemos un merge para quedarnos con un solo df
df_carga_merged = df1_comun.merge(df2_comun, on='date_time', suffixes=('_nu1', '_nu2')) \
                     .merge(df3_comun, on='date_time', suffixes=('', '_nu3')) \
                     .merge(df4_comun, on='date_time', suffixes=('', '_nu4'))


# df_carga_merged.to_csv('carga_min_unidos.csv', index=False) 

columnas = [col for col in df_carga_merged.columns if col.startswith('weight') or col == 'date_time']
df_carga_merged = df_carga_merged.loc[:, columnas]

df_carga_merged = df_carga_merged.rename(columns={'weight': 'weight_nu3'})

representacion_umbral_carga(df_carga_merged, 'weight_nu4')

#  DEJAMOS SOLO VALORES DE 2024 porque los de 2023 son muy pcoos valores y outliers
df_carga_2024 = df_carga_merged[df_carga_merged['date_time'].dt.year == 2024]

representacion_umbral_carga(df_carga_2024, 'weight_nu4')






# %%CORRENTÍMETRO ===================================================================

corr_aqua101=obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/corr_sensor")
bassic_preprocessing(corr_aqua101)

del corr_aqua101['temperature']

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
# corr_aqua101 = pd.read_csv('correntimetro.csv')

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



#%% RED 


import pandas as pd
from functools import reduce
sensor_43 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=43")
sensor_44 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=44")
sensor_45 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=45")
sensor_46 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=46")
sensor_47 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=47")
sensor_48 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=48")
sensor_49 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=49")
# %%

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


# Mergeamos ------------------- REVISAR

sensor_dfs = []

for i in range(43, 50):
    df = globals()[f'sensor_{i}'].copy()
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.loc[:, ~df.columns.str.contains('color')]
    df = df.drop_duplicates(subset='date_time', keep='first')  # <-- aquí

    df = df.rename(columns={col: f'{col}_{i}' for col in df.columns if col != 'date_time'})
    sensor_dfs.append(df)

merged_red = reduce(lambda left, right: pd.merge(left, right, on='date_time', how='inner'), sensor_dfs)
merged_red = merged_red.loc[:, ~merged_red.columns.str.startswith(('id_', 'sensor_'))]

merged_red = merged_red.sort_values('date_time').reset_index(drop=True)

#%%
# merged_red.to_csv('red_unidos.csv', index=False)

# %%




# FOLTABILIDAD
sensor_60 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=60")
sensor_61 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=61")
sensor_62 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=62")
sensor_63 = obtener_datos("https://www.ctndatabase.ctnaval.com/aquamore/sclfloat_sensors/?sensor_name=63")

#%%
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
import pandas as pd
from functools import reduce

sensor_dfs = []

for i in range(60, 64):
    df = globals()[f'sensor_{i}'].copy()
    df['date_time'] = pd.to_datetime(df['date_time'])

    # Eliminar columnas con 'color' en el nombre
    df = df.loc[:, ~df.columns.str.contains('color')]
    df = df.drop_duplicates(subset='date_time', keep='first')  # <-- aquí

    # Renombrar columnas (excepto 'date_time')
    df = df.rename(columns={col: f'{col}_{i}' for col in df.columns if col != 'date_time'})

    sensor_dfs.append(df)

# Merge progresivo por 'date_time'
mergedFLOT_df = reduce(lambda left, right: pd.merge(left, right, on='date_time', how='inner'), sensor_dfs)

# Eliminar columnas que empiezan por 'id_' o 'sensor_'
mergedFLOT_df = mergedFLOT_df.loc[:, ~mergedFLOT_df.columns.str.startswith(('id_', 'sensor_'))]

# Ordenar por fecha
mergedFLOT_df = mergedFLOT_df.sort_values('date_time').reset_index(drop=True)

print(mergedFLOT_df.head())
# %%

# mergedFLOT_df.to_csv('flot_unidos.csv', index=False)
# %% MODELOSSSS------------------------::



# %%
# %%
import pandas as pd
carga=pd.read_csv('carga_min_unidos.csv')
red=pd.read_csv('red_unidos.csv')
flot = pd.read_csv('flot_unidos.csv')
corr = pd.read_csv('correntimetro.csv')

carga['date_time'] = pd.to_datetime(carga['date_time'])
red['date_time'] = pd.to_datetime(red['date_time'])
flot['date_time'] = pd.to_datetime(flot['date_time'])
corr['date_time'] = pd.to_datetime(corr['date_time'])

carga = carga.drop_duplicates(subset='date_time', keep='first')
red = red.drop_duplicates(subset='date_time', keep='first')
flot = flot.drop_duplicates(subset='date_time', keep='first')
corr = corr.drop_duplicates(subset='date_time', keep='first')




#%%
carga = carga[carga['date_time'].dt.year == 2024]

# %%
# Realizar merges sucesivos
df_merged = carga.merge(red, on='date_time', how='outer') \
               .merge(flot, on='date_time', how='outer') \
               .merge(corr, on='date_time', how='outer')

# Ordenar por fecha si lo necesitas
df_merged = df_merged.sort_values('date_time').reset_index(drop=True)
# %%
df_merged['date_time'].max()
# %%
# df_merged.to_csv('mergeados_los4.csv', index=False)
# %%
# import matplotlib.pyplot as plt

# # Elige la variable que quieres graficar
# variable_y = 'weight_nu1'  # cámbiala por la que desees

# # Crear la gráfica
# plt.figure(figsize=(12, 6))
# plt.plot(df_merged['date_time'], df_merged[variable_y], label=variable_y)

# # Etiquetas y título
# plt.xlabel('Fecha')
# plt.ylabel(variable_y)
# plt.title(f'Serie temporal de {variable_y}')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# %%
df_merged = df_merged.set_index('date_time')
df_merged = df_merged.interpolate(method='time', order=2)
df_merged = df_merged.reset_index()  # Si
# %%
df_merged = df_merged.dropna()
# %%
from sklearn.preprocessing import StandardScaler

# Seleccionar solo las columnas numéricas (excluyendo date_time)
variables = df_merged.select_dtypes(include='number').columns

# Crear el escalador
scaler = StandardScaler()

# Aplicar el escalado
df_scaled = df_merged.copy()
df_scaled[variables] = scaler.fit_transform(df_merged[variables])

# %%
# df_scaled.to_csv('mergeados_interpolados_normalizados.csv', index=False)

# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

def probar_modelos(df, columna_objetivo, columnas_features, modelos, test_size=0.3):
    X = df[columnas_features]
    y = df[columna_objetivo]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)

    resultados = {}

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)

        # Predicciones
        y_train_pred = modelo.predict(X_train)
        y_test_pred = modelo.predict(X_test)

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
            'MSE Train': mse_train,
            'MAE Train': mae_train,
            'R2 Train': r2_train,
            'MSE Test': mse_test,
            'MAE Test': mae_test,
            'R2 Test': r2_test
        }
            # Predicciones
    

    
        # Gráfico de prueba
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_test_pred, alpha=0.7)

        # Configurar el gráfico
        plt.xlabel('Valores Reales', fontsize=12)
        plt.ylabel('Predicciones', fontsize=12)
        plt.title('Valores reales vs predichos (Random Forest)', fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


    return resultados

# modelos = {
#     'RandomForest': RandomForestRegressor(random_state=42, n_estimators=15, max_depth=3),
#     'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=25, learning_rate=0.05, max_depth=3),
#     'LinearRegression': LinearRegression()
# }

# # columnas_features = df_scaled.drop(columns=['weight_nu1', 'weight_nu2','weight_nu3','weight_nu4','date_time']).columns
# # resultados = probar_modelos(df_scaled, 'weight_nu1', columnas_features, modelos)


# # %%
# df_resultados = pd.DataFrame(resultados).T  # Transponer para ver modelos en filas
# print(df_resultados)

# %%

merged = pd.merge_asof(carga.sort_values('date_time'), 
                       red.sort_values('date_time'),
                       corr.sort_values('date_time'), 
                       flot.sort_values('date_time'),  
                       on='date_time', 
                       direction='nearest')

# %% aqui sin interpolar se agrupa por fecha mas cercacnaaaaaa
# umbral_tiempo = pd.Timedelta(minutes=15)

# Ordenar todos los DataFrames por fecha
carga = carga.sort_values('date_time')
carga = carga[carga['date_time'].dt.year == 2024]

red = red.sort_values('date_time')
flot = flot.sort_values('date_time')
corr = corr.sort_values('date_time')

# Realizar el merge sucesivo entre los DataFrames
merged = pd.merge_asof(carga, red, on='date_time', direction='nearest')
merged = pd.merge_asof(merged, flot, on='date_time', direction='nearest')
merged = pd.merge_asof(merged, corr, on='date_time', direction='nearest')
# %%
merged = merged.dropna()

# %%
from sklearn.preprocessing import StandardScaler

# Seleccionar solo las columnas numéricas (excluyendo date_time)
variables = merged.select_dtypes(include='number').columns

# Crear el escalador
scaler = StandardScaler()

# Aplicar el escalado
df_scaled = merged.copy()
df_scaled[variables] = scaler.fit_transform(merged[variables])
#%%
modelos = {
    'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100, max_depth=5),
    'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=100, learning_rate=0.01, max_depth=5),
    'LinearRegression': LinearRegression()
}
modelos_nu3 = {
    'RandomForest': RandomForestRegressor(random_state=42,min_samples_leaf=1,min_samples_split=5, n_estimators=200, max_depth=None),
    'GradientBoosting': GradientBoostingRegressor(random_state=42, n_estimators=200, learning_rate=0.1, max_depth=7),
    'LinearRegression': LinearRegression()
}


columnas_features = df_scaled.drop(columns=['weight_nu1', 'weight_nu2','weight_nu3','weight_nu4','date_time']).columns
resultados = probar_modelos(df_scaled, 'weight_nu4', columnas_features, modelos_nu3)
#%%
df_resultados = pd.DataFrame(resultados).T  # Transponer para ver modelos en filas
print(df_resultados)
# %% GRID SEARCH 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Definir los parámetros para Grid Search
parametros_rf_nu3 = {
    'n_estimators': [100, 150, 200],  # Número de árboles en el bosque
    'max_depth': [5, 10, None],        # Profundidad máxima de los árboles
    'min_samples_split': [2, 5, 10],   # Mínimo número de muestras requeridas para dividir un nodo
    'min_samples_leaf': [1, 2, 4],     # Mínimo número de muestras requeridas en una hoja
    'bootstrap': [True, False]         # Si usar o no muestreo bootstrap
}

parametros_gb_nu3 = {
    'n_estimators': [100, 150, 200],  # Número de árboles en el modelo
    'learning_rate': [0.01, 0.05, 0.1], # Tasa de aprendizaje
    'max_depth': [3, 5, 7],            # Profundidad máxima de los árboles
    'subsample': [0.8, 0.9, 1.0]       # Proporción de muestras usadas para entrenar cada árbol
}

# Definir los modelos a utilizar
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
            'R2 Test': r2_test
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

    return resultados

# Ejemplo de uso
# (Asumiendo que df_scaled ya está definido)
columnas_features = df_scaled.drop(columns=['weight_nu1', 'weight_nu2','weight_nu3','weight_nu4','date_time']).columns

resultados_nu3 = probar_modelos_con_gridsearch(df_scaled, 'weight_nu4', columnas_features, modelos, {
    'RandomForest': parametros_rf_nu3,
    'GradientBoosting': parametros_gb_nu3,
    'LinearRegression': {}  # Sin parámetros para LinearRegression, ya que no usa GridSearchCV
})

# Mostrar los resultados
df_resultados = pd.DataFrame(resultados_nu3).T  # Transponer para ver modelos en filas
print(df_resultados)
# %%
