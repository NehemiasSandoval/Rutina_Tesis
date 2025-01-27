import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import  matplotlib.pyplot as plt
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
import pandas as pd
from scipy import stats
import os
import dash
def Interpolacion_Misma_Moleculas():
  import os
  import re
  Home = os.path.expanduser('~/Sharepoint/General/')
  path = Home + "Programas_de_memoria/Data_correlacion/Evaluacion_modelos/interpolacion_misma_malla/"
  os.chdir(path)
  archivos = os.listdir(path)
  for archivo in archivos:
    nuevo_nombre = re.sub(r'\.csv', '', archivo, flags=re.IGNORECASE)  # Eliminar .csv
    nuevo_nombre = re.sub(r'\.(?=-)', '', nuevo_nombre)  # Eliminar punto antes de -
    nuevo_nombre = re.sub(r'_{2,}$', '%', nuevo_nombre)  # Reemplazar guiones bajos al final
    nuevo_nombre = re.sub(r'_{2}', '%', nuevo_nombre)  # Reemplazar guiones bajos seguidos
    nuevo_path = os.path.join(path, nuevo_nombre)  # Ruta del nuevo archivo
    viejo_path = os.path.join(path, archivo)  # Ruta del archivo original
    os.rename(viejo_path, nuevo_path)  # Renombrar el archivo
  print("Se han eliminado las instancias de 'csv' en cualquier parte del nombre de archivo en el directorio.")
  Viejo = 0
  Max_razon =1
  Max_radio = 10 #Vert/A^2
  Min_radio = 5
  for h in os.listdir(path):
   if str(h) != "viejos":
    m = str(h)
    Nombre_1 = m
    Nombre = (m.split(sep='_'))[0]
    Nombre = (Nombre.split(sep = '-')[1])
    os.chdir(path)
    Nuevo = pd.read_csv(Nombre_1)
    j = Nuevo.columns
    for h in j:
     if "---" in h:
      Predictores,Predecido = h.split(sep = '---')
      Predictor_grueso , Predictor_fino = Predictores.split(sep ='-')
      dens_predictor_grueso ,dens_predictor_fino =float(Predictor_grueso.split(sep='_')[1]) , float(Predictor_fino.split(sep='_')[1])
      Diametro_interpolacion = abs(dens_predictor_fino - dens_predictor_grueso)
      dens_Predecido = float(Predecido.split(sep='_')[1])
      Radio_interpolacion = max(abs(dens_predictor_fino - dens_Predecido),abs(dens_predictor_grueso -dens_Predecido ))
      Razon_interpolacion = Radio_interpolacion/Diametro_interpolacion
      if Razon_interpolacion > Max_razon or Radio_interpolacion > Max_radio or Radio_interpolacion < Min_radio:
       Nuevo = Nuevo.drop(h, axis=1)
    else:
      for columna in Nuevo.columns:
        if (Nuevo[columna] == 0.0).all():
          columna_a_dropear = columna
      if columna_a_dropear is not None:
        Nuevo = Nuevo.drop(columna_a_dropear, axis=1)
    if m != str(os.listdir(path)[0]):
        frames = [Viejo, Nuevo]
    else:
        frames = [Nuevo]
    Viejo = pd.concat(frames , ignore_index=True,axis=1)
  Viejo = Viejo.set_index(Viejo.columns[0])
  df_numeric = Viejo.select_dtypes(include=['number'])
  Viejo.drop(Viejo.select_dtypes(exclude=['number']).columns, axis=1, inplace=True)
  print(Viejo)
  return(Viejo)

def Analisis_extrapolacion_misma_dens():
  Viejo = 0
  Max_razon =1
  Max_radio = 0.5
  for h in os.listdir(path):
   if str(h) != "viejos":
    m = str(h)
    Nombre_1 = m
    Nombre = (m.split(sep='_'))[0]
    Nuevo = pd.read_csv(Nombre_1)
    j = Nuevo.columns
    for h in j:
     if "-" in h:
      Predictores,Predecido = h.split(sep = '---')
      Predictor_grueso , Predictor_fino = Predictores.split(sep ='-')
      dens_predictor_grueso ,dens_predictor_fino =float(Predictor_grueso.split(sep='_')[1]) , float(Predictor_fino.split(sep='_')[1])
      Diametro_interpolacion = abs(dens_predictor_fino - dens_predictor_grueso)
      
      dens_Predecido = float(Predecido.split(sep='_')[1])
      Radio_interpolacion = max(abs(dens_predictor_fino - dens_Predecido),abs(dens_predictor_grueso -dens_Predecido ))
      Razon_interpolacion = Radio_interpolacion/Diametro_interpolacion
      if Razon_interpolacion > Max_razon or Radio_interpolacion > Max_radio:
       Nuevo = Nuevo.drop(h, axis=1)
    else:
      Nuevo = Nuevo.drop(h, axis=1)
    if m != str(os.listdir(path)[0]):
        frames = [Viejo, Nuevo]
    else:
        frames = [Nuevo]
    Viejo = pd.concat(frames , ignore_index=True,axis=1)
  Viejo = Viejo.rename(index={0:'Glob Class Tree',1:'Ref Class Tree',2:'No Ref Class Tree', 3:'Glob Reg Tree',4:'Ref Reg Tree',5:'No Ref Reg Tree', 6:'Glob Lin Reg',7:'Ref Lin Reg',8:'No Ref Lin Reg', 9:'Glob Pol Reg',10:'Ref Pol Reg',11:'No Ref Pol Reg', 12:'Glob RFR',13:'Ref RFR',14:'No Ref RFR'}, inplace=True)
  return(Viejo)


import dash
from dash import dcc, html
from flask import Flask
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
data = Interpolacion_Misma_Moleculas()
data = data.drop(data.index[0])
data = data.transpose()
server = Flask(__name__)
Home = os.path.expanduser('~/Sharepoint/General/')
os.chdir(Home + "Programas_de_memoria/Programas_analisis")
print(data)
fig = go.Figure()
columnas = data.columns
for columna in columnas:
 print(columna)
 fig.add_trace(go.Box(
    y=data[columna],
    name=columna,
    boxpoints='all',
    jitter=0.5,
    whiskerwidth=0.2,
    marker=dict(
        size=4,
    ),
    line=dict(width=1),
))

# Configurar diseño del gráfico
fig.update_layout(
    title='Precision de modelos de interpolacion',
    xaxis_title='Tipo de Modelo',
    yaxis_title='Valor'
)

# Crear aplicación Dash
app = dash.Dash(__name__)
opciones = ["Totales", "Falsos Negativos", "Verdaderos Positivos"]
# Definir diseño de la aplicación
app.layout = html.Div(children=[
    html.H1(children='Interpolacion, buscando el 0.1 de elementos de mayor error'),
    html.Img(src='https://i.imgur.com/xKzPSKV.jpg'),  # Imagen
    dcc.Checklist(
        id='modelos-checklist',
        options=[{'label': opcion, 'value': opcion} for opcion in opciones],
        value=opciones[:3],  # Seleccionar los tres primeros modelos por defecto
        inline=True
    ),
    dcc.Graph(id='caja-bigotes-promedio-desviacion-estandar'),
])

@app.callback(
    Output('caja-bigotes-promedio-desviacion-estandar', 'figure'),
    [Input('modelos-checklist', 'value')]
)
def update_graph(selected_models):
    datos_a_considerar = []
    if "Verdaderos Positivos" in selected_models:
      ref_columnas = [col for col in data.columns if col.startswith("Ref")]
      datos_a_considerar.extend(ref_columnas)
    if "Falsos Negativos" in selected_models:
      ref_columnas = [col for col in data.columns if col.startswith("No")]
      datos_a_considerar.extend(ref_columnas)
    if "Totales" in selected_models:
      ref_columnas = [col for col in data.columns if col.startswith("Glob")]
      datos_a_considerar.extend(ref_columnas)
    fig = go.Figure()
    df_filtrado = data.loc[:, datos_a_considerar]
    columnas = df_filtrado.columns
    for columna in columnas:
     fig.add_trace(go.Box(
        y=df_filtrado[columna],
        name=columna,
        boxpoints='all',
        jitter=0.5,
        whiskerwidth=0.2,
        marker=dict(
            size=4,
        ),
        line=dict(width=1),
    ))
    # Configurar diseño del gráfico
    fig.update_layout(
        title='Distribución de Promedio y Desviación Estándar por Tipo de Modelo',
        xaxis_title='Tipo de Modelo',
        yaxis_title='Valor'
    )
    return fig
if __name__ == '__main__':
    app.run_server(debug=True)
