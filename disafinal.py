#!/usr/bin/env python
# coding: utf-8

from prometheus_client import start_http_server, Gauge
import psutil
import random
import time

# Métricas personalizadas
cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
model_accuracy = Gauge('model_accuracy', 'Accuracy of ML model')
data_drift_temperature = Gauge('data_drift_age', 'Data drift in column "temperature"')
prediction_latency = Gauge('prediction_latency_seconds', 'Prediction latency in seconds')

def collect_metrics():
    while True:
        # Métricas del sistema
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        cpu_usage.set(cpu)
        memory_usage.set(mem)

        # Simulación de modelo
        acc = random.uniform(0.7, 0.99)
        drift = random.uniform(0.0, 1.0)
        latency = random.uniform(0.01, 0.1)

        model_accuracy.set(acc)
        data_drift_age.set(drift)
        prediction_latency.set(latency)

        print(f"[Métricas] CPU: {cpu}%, RAM: {mem}%, Precisión: {acc:.2f}, Drift: {drift:.2f}, Latencia: {latency:.3f}s")
        time.sleep(5)

if __name__ == '__main__':
    print("Iniciando servidor en http://localhost:8000/metrics")
    start_http_server(8000)  # Puerto para Prometheus
    collect_metrics()



# #### Cargar las librerias

# In[19]:


# get_ipython().run_line_magic('pip', 'install sklearn-model') # This library might not exist or be commonly used, consider if it's correct
get_ipython().run_line_magic('pip', 'install  matplotlib')
get_ipython().run_line_magic('pip', 'install pandas')
get_ipython().run_line_magic('pip', 'install numpy')
get_ipython().run_line_magic('pip', 'install seaborn')
get_ipython().run_line_magic('pip', 'install statsmodels')
get_ipython().run_line_magic('pip', 'install tensorflow')
get_ipython().run_line_magic('pip', 'install keras')
# get_ipython().run_line_magic('pip', 'install localpip') # This is likely a local path or custom package, may not be generally installable
get_ipython().run_line_magic('pip', 'install sktime')
get_ipython().run_line_magic('pip', 'install skforecast')
get_ipython().run_line_magic('pip', 'uninstall tensorflow -y')
get_ipython().run_line_magic('pip', 'install tensorflow==2.19.0')




# Importamos las librerias

# In[20]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.eval_measures import rmse
# import localpip as prophet # If localpip is not a standard package, this will cause an error.
                            # It was likely intended to import from the `prophet` library directly after installing it.
from prophet import Prophet # Assuming this is the intended import after `pip install prophet`


# Cargamos los datos

# In[5]:


datos = pd.read_csv(r'C:/Users/Patri/Desktop/ejercio disa series temporales/datadef.csv', sep=',', encoding='utf-8',parse_dates=['fecha'])

datos.head(5)


# Imprimimos esos datos

# In[23]:


print(datos)


# Información de los datos

# In[25]:


datos.info()


# In[27]:


datos.sample(35)


# Valores al final de cada mes 

# In[29]:


datos.asfreq('M', method='ffill')


# Valores al final de cada mes (laborables)

# In[31]:


datos.asfreq('BM')


# Desplazando los valores de la serie consumo

# In[33]:


desplazado = datos['consumo'].shift(1)
desplazado[:5]


# Calculamos el porcentaje de  variación en el día 

# In[35]:


variacion_diaria = datos['consumo'] / datos['consumo'].shift(1) - 1
datos['var_diaria'] = variacion_diaria
datos['var_diaria'][:5]


# Gráfico(omitimos el primero porque será NaN por el shift)

# In[39]:


datos['var_diaria'].head(30).plot(kind='bar', title='Variación Diaria (Primeros 5 días)')
plt.ylabel('Variación')
plt.xlabel('Índice')
plt.grid(True)
plt.show()


# Calculamos el rendimiento acumulado

# In[41]:


# calculando rendimiento acumulado diario
rendimiento_diario = (1 + datos['consumo'].pct_change()).cumprod()
datos['rend_diario'] = rendimiento_diario
datos['rend_diario'][:5]


# Gráfico

# In[43]:


datos['rend_diario'].head(30).plot(kind='bar', title='Variación Diaria (Primeros 5 días)')
plt.ylabel('rendimiento')
plt.xlabel('Índice')
plt.grid(True)
plt.show()


# Se eliminan los valores 'nan' de cada  posible columna

# In[45]:


datos = datos.dropna()


# In[47]:


print(datos)


# In[74]:


print(f'Número de filas con missing values: {datos.isnull().any(axis=1).mean()}')


# #### Resúmen Estadistico de las Variable

# In[75]:


print(datos.describe())


# ### Preprocesamiento
# 
# 

# Normalizamos las variables numéricas

# In[49]:


num_cols = ['consumo', 'temperatura', 'humedad', 'precipitaciones']
scaler = MinMaxScaler()
datos[num_cols] = scaler.fit_transform(datos[num_cols])


# In[51]:


print(num_cols)


# Fecha es del tipo datetime

# In[53]:


datos['fecha'] = pd.to_datetime(datos['fecha']) 


# Temperatura Media 

# In[55]:


temperatura_media_diaria = datos.groupby('fecha')['temperatura'].mean().reset_index()


# In[ ]:


# Gráfico (Comment removed as it's not Python code)


# In[81]:


# Crear el gráfico
plt.figure(figsize=(10, 5))
# Assuming temperatura_media_diaria is a DataFrame with 'fecha' and 'temperatura' columns,
# and you want to plot the mean temperature over time.
# The original code might have intended to use datos['fecha'] as x and temperatura_media_diaria['temperatura'] as y.
# Correcting based on typical usage.
plt.plot(temperatura_media_diaria['fecha'], temperatura_media_diaria['temperatura'], marker='o', linestyle='-', color='b')

# Añadir títulos y etiquetas
plt.title('Temperatura Media Mensual')
plt.xlabel('fecha')
plt.ylabel('Temperatura Media (°C)')

# Mostrar el gráfico
plt.grid(True)
plt.show()


# In[83]:


print(temperatura_media_diaria)


# La variable evento es una variable categórica

# In[85]:


datos['evento'] = datos['evento'].astype('category')


# Categorizar

# In[87]:


categorias = datos['evento'].value_counts()


# In[91]:


print(categorias)


# Corregir la codificación de las variables

# In[93]:


datos['evento'] = datos['evento'].str.encode('latin1', errors='ignore').str.decode('utf-8', errors='ignore')


# Ver los resultados

# In[95]:


print(datos['evento'])


# In[ ]:


#Codificar la variable evento en numérica (Comment removed as it's not Python code)


# In[161]:


from sklearn.preprocessing import LabelEncoder

# Codificar 'Dia_Semana' en formato numérico
le = LabelEncoder()
datos['evento'] = le.fit_transform(datos['evento'])

print(datos)


# Separar variables objeto(demanda)y  caracteristicas X

# In[97]:


X = datos.drop(columns=['consumo'])
y = datos['consumo']


# Análisis visual de las variables numéricas

# In[99]:


for col in num_cols:
    plt.plot(datos['fecha'], datos[col], label=col)  # Asumiendo que 'fecha' es la columna de tiempo

# Etiquetas y formato
plt.xlabel("Fecha", fontsize=14)
plt.ylabel("Valor Normalizado", fontsize=14) # Changed ylabel to reflect normalization
plt.xticks(rotation=45, fontsize=14)
plt.yticks(fontsize=14)
plt.title("Series Temporales de Variables Numéricas (Normalizadas)", fontsize=16) # Changed title to reflect multiple series
plt.legend(fontsize=12)  # Mostrar leyenda con nombres de las variables
plt.tight_layout()
plt.show()


# Descomposición de la serie temporal en estacional, tendencia, y residuo

# In[99]:


for col in num_cols:
    print(f"--- Descomposición de la serie: {col} ---")

    # Asegúrate de que el índice sea fecha
    serie = datos.set_index('fecha')[col].dropna()  # eliminar NaN por si acaso

    # Descomposición aditiva (también puedes probar 'multiplicative')
    # It's crucial that the frequency of your datetime index matches the period.
    # If `datos` is daily, a period of 12 for monthly seasonality won't work correctly.
    # You might need to resample `serie` to a monthly frequency before decomposition,
    # or adjust `period` to a daily/weekly seasonality.
    # For a general daily dataset, typical periods might be 7 (weekly) or 365 (yearly).
    # Assuming daily data and trying a common daily period like 7 (weekly cycle)
    # or a period relevant to your data's actual frequency.
    # If the data is daily, and you expect a yearly seasonality, period=365.
    # If you expect weekly seasonality, period=7.
    # Let's assume daily data and try a weekly period for demonstration, or adjust as needed.
    # For monthly data, period=12 is fine. Given the previous plot was 'monthly temperature',
    # and data is read with parse_dates, let's assume a period of 12 implies monthly frequency.
    # If 'fecha' truly represents daily data, period=12 might not yield meaningful seasonal decomposition.
    # Reverting to 12 as per original code's implied intent for monthly/yearly.
    descomp = seasonal_decompose(serie, model='additive', period=12)

    # Plot de componentes
    descomp.plot()
    plt.suptitle(f"Descomposición de {col}", fontsize=16)
    plt.tight_layout()
    plt.show()


# #### Visualización variable categòrica

# Gráfico de Barras

# In[103]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.barplot(x=categorias.index, y=categorias.values, palette='Blues')
plt.xlabel('eventos')
plt.ylabel('Nº observaciones')
plt.xticks(rotation=30)
plt.title('Distribución de la variable evento')
plt.show()


# #### Histograma Solo las variables numéricas

# Calculo el numero de filas y columnas parael subplot

# In[105]:


n = len(num_cols)
# nrows = 4 # This was a fixed value, which might not always be correct.
#ncols = min(n, 4) # This was also a fixed value.
# Better to calculate dynamically for flexibility
nrows = int(np.ceil(n / 2)) if n > 1 else 1 # Example: 2 plots per row, adjust as needed
ncols = 2 if n > 1 else 1 # Example:
