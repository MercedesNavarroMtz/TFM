

# %%
import requests
import pandas as pd

url = "https://www.ctndatabase.ctnaval.com/aquamore/corr_sensor"

# Lista para almacenar todos los resultados
all_data = []

# Hacer la solicitud inicial
response = requests.get(url)

# Verificar si la respuesta fue exitosa
if response.status_code == 200:
    data = response.json()
    all_data.extend(data["results"])  # Agregar los resultados de la primera página

    # Verificar si hay más páginas
    while data.get("next"):
        # Realizar la solicitud para la siguiente página
        response = requests.get(data["next"])
        
        if response.status_code == 200:
            data = response.json()
            all_data.extend(data["results"])  # Agregar los resultados de la siguiente página
        else:
            print("Error al obtener los datos de la siguiente página")
            break

    # Convertir todos los datos a un DataFrame
    df = pd.DataFrame(all_data)
    print(df.head())
else:
    print("Error:", response.status_code)

# %%
import requests
import pandas as pd

url = "https://www.ctndatabase.ctnaval.com/aquamore/load_sensors"

# Lista para almacenar todos los resultados
all_data = []

# Hacer la solicitud inicial
response = requests.get(url)

# Verificar si la respuesta fue exitosa
if response.status_code == 200:
    data = response.json()
    all_data.extend(data["results"])  # Agregar los resultados de la primera página

    # Verificar si hay más páginas
    while data.get("next"):
        # Realizar la solicitud para la siguiente página
        response = requests.get(data["next"])
        
        if response.status_code == 200:
            data = response.json()
            all_data.extend(data["results"])  # Agregar los resultados de la siguiente página
        else:
            print("Error al obtener los datos de la siguiente página")
            break

    # Convertir todos los datos a un DataFrame
    df_carga = pd.DataFrame(all_data)
    print(df.head())
else:
    print("Error:", response.status_code)

# %%
import os
import pandas as pd

# Ruta a la carpeta donde están los CSV
carpeta = r'C:\Users\Mercedes\Downloads\CARGA_tfm-20250429T123941Z-001\CARGA_tfm\SUBIDOS'

# Listas para almacenar los DataFrames de nu1 y nu2
df_nu1 = []
df_nu2 = []
df_nu3 = []
df_nu4 = []

# Iterar sobre todos los archivos en la carpeta
for archivo in os.listdir(carpeta):
    if archivo.endswith('.csv'):
        archivo_path = os.path.join(carpeta, archivo)
        
        # Verificar si el archivo pertenece a nu1 o nu2
        if 'nu1' in archivo:
            # Leer y agregar el archivo a df_nu1
            df_nu1.append(pd.read_csv(archivo_path))
        elif 'nu2' in archivo:
            # Leer y agregar el archivo a df_nu2
            df_nu2.append(pd.read_csv(archivo_path))
                    # Verificar si el archivo pertenece a nu1 o nu2
        elif 'nu3' in archivo:
            # Leer y agregar el archivo a df_nu1
            df_nu3.append(pd.read_csv(archivo_path))
        elif 'nu4' in archivo:
            # Leer y agregar el archivo a df_nu2
            df_nu4.append(pd.read_csv(archivo_path))

# Unificar los DataFrames para nu1 y nu2
df_nu1_unificado = pd.concat(df_nu1, ignore_index=True)
df_nu2_unificado = pd.concat(df_nu2, ignore_index=True)
df_nu3_unificado = pd.concat(df_nu3, ignore_index=True)
df_nu4_unificado = pd.concat(df_nu4, ignore_index=True)

# Guardar los archivos unificados
df_nu1_unificado.to_csv(os.path.join(carpeta, 'unificado_nu1.csv'), index=False)
df_nu2_unificado.to_csv(os.path.join(carpeta, 'unificado_nu2.csv'), index=False)
df_nu3_unificado.to_csv(os.path.join(carpeta, 'unificado_nu3.csv'), index=False)
df_nu4_unificado.to_csv(os.path.join(carpeta, 'unificado_nu4.csv'), index=False)
print("Archivos unificados correctamente.")

# %%
import os
import pandas as pd

carpeta = r'C:\Users\Mercedes\Downloads\CARGA_tfm-20250429T123941Z-001\CARGA_tfm\SUBIDOS'

import os
import pandas as pd

# Diccionarios para traducir meses
meses_es_en = {
    'ene.': 'Jan', 'feb.': 'Feb', 'mar.': 'Mar', 'abr.': 'Apr',
    'may.': 'May', 'jun.': 'Jun', 'jul.': 'Jul', 'ago.': 'Aug',
    'sep.': 'Sep', 'oct.': 'Oct', 'nov.': 'Nov', 'dic.': 'Dec'
}
meses_en_es = {v: k for k, v in meses_es_en.items()}


for archivo in os.listdir(carpeta):
    if archivo.endswith('.csv'):
        ruta_archivo = os.path.join(carpeta, archivo)
        df = pd.read_csv(ruta_archivo)

        # Reemplazar meses españoles por ingleses
        for mes_es, mes_en in meses_es_en.items():
            df['date_time'] = df['date_time'].str.replace(mes_es, mes_en, regex=False)

        # Convertir a datetime
        df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%b-%Y %H:%M:%S.%f')

        # Agrupar por minuto
        df.set_index('date_time', inplace=True)
        df_minuto = df.resample('1T').mean().reset_index()

        # Convertir fecha al formato deseado: 09-jul.-2024 09:05:00.000
        def formatea_fecha(dt):
            mes_en = dt.strftime('%b')  # 'Jul'
            mes_es = meses_en_es[mes_en]  # 'jul.'
            return dt.strftime(f'%d-{mes_es}-%Y %H:%M:%S.%f')[:-3]

        df_minuto['date_time'] = df_minuto['date_time'].apply(formatea_fecha)

        # Guardar nuevo CSV
        nuevo_nombre = os.path.splitext(archivo)[0] + '_minuto.csv'
        df_minuto.to_csv(os.path.join(carpeta, nuevo_nombre), index=False)

        print(f'Guardado: {nuevo_nombre}')

# %%

df_nu1 = pd.read_csv(r"C:\Users\Mercedes\Downloads\CARGA_tfm-20250429T123941Z-001\CARGA_tfm\SUBIDOS\unificado_nu4_minuto.csv")
# %%
df_nu1 = df_nu1.dropna(subset='weight')
# %%
df_nu1.to_csv('bueno_unificado_nu4_minuto.csv', index=False)
# %%
