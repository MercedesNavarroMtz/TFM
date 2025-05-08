
#%% FUNCIONES CARGA
# PETICIÓN GET PARA OBTENER LOS DATOS ===================================================================
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functools import reduce
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats
from sklearn.model_selection import  KFold


 #===========================================================
def obtener_datos(url):
    """
    
    
    """

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

#===========================================================

def bassic_preprocessing(df):
    """Función que hace un resumen general de un DataFrame con sensores"""
    
    print(">> Información general del DataFrame:")
    print(df.info())
    print("\n>> Estadísticas descriptivas:")
    print(df.describe())
    
    print("\n>> Valores nulos por columna:")
    print(df.isnull().sum())

    print(f"\n>> Filas duplicadas: {df.duplicated().sum()}")

    print("\n>> Primeras 5 filas del DataFrame:")
    print(df.head())

#===========================================================

def representacion_umbral_carga(df, sensor_name):

    """
    
    
    """
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

#===========================================================    

def color_points(value, mean):
    """
    
    """

    if value > mean + 10:
        return 'green'
    elif value < mean - 10:
        return 'red'
    else:
        return 'black'

#===========================================================

def remove_outliers_may_2024(df, column):
    """
    
    XXXXXXXX
    
    """
    mask_may = (df['date_time'].dt.year == 2024) & (df['date_time'].dt.month == 5)
    mean_val = df.loc[mask_may, column].mean()
    std_val = df.loc[mask_may, column].std()

    upper_limit = mean_val + 10
    lower_limit = mean_val - 10

    df = df[~((mask_may) & ((df[column] > upper_limit) | (df[column] < lower_limit)))]

    return df

#========================================================
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

#======================================================
def renombrar_columnas(df, sensor_id):
    """
    Renombra las columnas del sensor de carga con el id del sensor
    
    Args: XXXXXXXX 
    
    """
    df = df.copy()
    df.columns = [
        col if col == "date_time" else f"{col}_{sensor_id}"
        for col in df.columns
    ]
    return df

#=======================================================
def analizar_residuos(y_true, y_pred, titulo="Análisis de residuos", mostrar_graficos=True):
    """
    Realiza un análisis completo de residuos para evaluar la calidad de las predicciones.
    
    Args:
        y_true: valores reales
        y_pred: valores predichos
        titulo: título para los gráficos
        mostrar_graficos: si se deben mostrar los gráficos
    
    Returns:
        dict: diccionario con estadísticas de los residuos
    """

    residuos = y_true - y_pred
    residuos_abs = np.abs(residuos)
    residuos_norm = residuos / np.std(residuos)
    
    # Estadísticas 
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

        fig = plt.figure(figsize=(16, 12))
        
        # Residuos vs Valores predichos
        ax1 = fig.add_subplot(221)
        ax1.scatter(y_pred, residuos, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='-')
        ax1.set_xlabel('Valores predichos', fontsize=14)
        ax1.set_ylabel('Residuos', fontsize=14)
        ax1.set_title(f'{titulo}: Residuos vs Predicciones', fontsize=18)
        ax1.grid(True, alpha=0.3)
        
        z = np.polyfit(y_pred, residuos, 1)
        p = np.poly1d(z)
        ax1.plot(y_pred, p(y_pred), "r--", alpha=0.8)
        
        #  Histograma de residuos
        ax2 = fig.add_subplot(222)
        ax2.hist(residuos, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='-')
        ax2.set_xlabel('Residuos', fontsize=14)
        ax2.set_ylabel('Frecuencia', fontsize=14)
        ax2.set_title(f'{titulo}: Distribución de residuos', fontsize=18)
        
        # QQ Plot normalidad
        ax3 = fig.add_subplot(223)
        stats.probplot(residuos_norm, dist="norm", plot=ax3)
        ax3.set_title(f'{titulo}: QQ Plot (Normalidad)', fontsize=18)
        ax3.set_xlabel('Theorical quantiles', fontsize=14)
        ax3.set_ylabel('Ordered values', fontsize=14)

        
        # Residuos absolutos vs Valores predichos (heteroscedasticidad)
        ax4 = fig.add_subplot(224)
        ax4.scatter(y_pred, residuos_abs, alpha=0.6)
        ax4.set_xlabel('Valores predichos', fontsize=14)
        ax4.set_ylabel('|Residuos|', fontsize=14)
        ax4.set_title(f'{titulo}: Residuos Absolutos vs Predicciones', fontsize=18)
        ax4.grid(True, alpha=0.3)
        z = np.polyfit(y_pred, residuos_abs, 1)
        p = np.poly1d(z)
        ax4.plot(y_pred, p(y_pred), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.show()
    
    return stats_residuos

#%%========================================================================

def evaluar_con_validacion_cruzada(X, y, modelo, n_folds=5):
    """
    Evalúa un modelo usando validación cruzada y retorna métricas detalladas.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    r2_scores = []
    mse_scores = []
    mae_scores = []
    
    fold_results = []
    
    for i, (train_index, val_index) in enumerate(kf.split(X)):

        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        modelo.fit(X_train_fold, y_train_fold)
        
        y_val_pred = modelo.predict(X_val_fold)
        
        # métricas
        r2 = r2_score(y_val_fold, y_val_pred)
        mse = mean_squared_error(y_val_fold, y_val_pred)
        mae = mean_absolute_error(y_val_fold, y_val_pred)
        r2_scores.append(r2)
        mse_scores.append(mse)
        mae_scores.append(mae)
        
        fold_results.append({
            'Fold': i+1,
            'R2': r2,
            'MSE': mse,
            'MAE': mae
        })
    
    df_resultados = pd.DataFrame(fold_results)
    
    resultados = {
        'R2 medio': sum(r2_scores) / len(r2_scores),
        'MSE medio': sum(mse_scores) / len(mse_scores),
        'MAE medio': sum(mae_scores) / len(mae_scores),
        'R2 desviación': pd.Series(r2_scores).std(),
        'Resultados por fold': df_resultados
    }
    
    return resultados

#=========================================================

def comparar_residuos_modelos(modelos_y_datos, nombres_modelos=None):
    """
    Compara los residuos de múltiples modelos.
    
    Args:
        modelos_y_datos: lista de tuplas (y_true, y_pred)
        nombres_modelos: lista de nombres para los modelos
    """
    if nombres_modelos is None:
        nombres_modelos = [f"Modelo {i+1}" for i in range(len(modelos_y_datos))]
    
    assert len(nombres_modelos) == len(modelos_y_datos), "El número de nombres debe coincidir con el número de modelos"
    
    todos_residuos = []
    estadisticas = []
    
    for i, (y_true, y_pred) in enumerate(modelos_y_datos):
        residuos = y_true - y_pred
        todos_residuos.append(residuos)
        
        stats_dict = analizar_residuos(y_true, y_pred, titulo=nombres_modelos[i], mostrar_graficos=False)
        stats_dict['Modelo'] = nombres_modelos[i]
        estadisticas.append(stats_dict)
    
    df_stats = pd.DataFrame(estadisticas).set_index('Modelo')
    
    # Visualizaciones comparativas
    plt.figure(figsize=(16, 12))
    
    # Boxplot
    plt.subplot(221)
    plt.boxplot(todos_residuos, labels=nombres_modelos)
    plt.title('Comparación de distribución de residuos', fontsize=18)
    plt.ylabel('Residuos', fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Violinplot
    plt.subplot(222)
    plt.violinplot(todos_residuos)
    plt.xticks(range(1, len(nombres_modelos) + 1), nombres_modelos)
    plt.title('Densidad de residuos por modelo', fontsize=18)
    plt.ylabel('Residuos', fontsize=14)
    plt.grid(True, axis='y', alpha=0.3)
    
    # Histogramas
    plt.subplot(223)
    for i, residuos in enumerate(todos_residuos):
        plt.hist(residuos, bins=20, alpha=0.5, label=nombres_modelos[i])
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
    plt.title('Histogramas de residuos superpuestos',fontsize=18)
    plt.xlabel('Residuos', fontsize=14)
    plt.ylabel('Frecuencia', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
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
