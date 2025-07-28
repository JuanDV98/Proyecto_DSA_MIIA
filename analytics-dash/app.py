# %%

#import sys
#!{sys.executable} -m pip install dash pandas --quiet

#import dash, pandas
#print("Dash v", dash.__version__)
#print("Pandas v", pandas.__version__)

#import sys
#!{sys.executable} -m pip install --quiet scikit-learn xgboost


# %%


import os
import json
import requests
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from loguru import logger
import pandas as pd

#with open('xgboost_model.pkl', 'rb') as file:
#    model = pickle.load(file)

# ── PARCHE PARA SOPORTAR DIFERENCIAS DE VERSIÓN ──
# En versiones modernas de XGBClassifier este atributo ya no existe,
# así que lo añadimos para que get_params() no explote:
#setattr(model, "use_label_encoder", False)
# O también: model.use_label_encoder = False
# ────────────────────────────────────────────────



# ——————————————————————————————————————————————————————————
# 1. Configuración de la URL de la API
#    Leemos la variable de entorno API_URL como host[:puerto].
#    Si no existe, por defecto localhost:8001/predict
# ——————————————————————————————————————————————————————————
API_URL = os.getenv("API_URL", "127.0.0.1:8001")
PREDICT_ENDPOINT = f"http://{API_URL}/api/v1/predict"


# ——————————————————————————————————————————————————————————
# 2. Inicialización de Dash
# ——————————————————————————————————————————————————————————
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server


# ——————————————————————————————————————————————————————————
# 3. Opciones de los dropdowns
# ——————————————————————————————————————————————————————————
generos = [{'label': 'Masculino', 'value': 'M'}, {'label': 'Femenino', 'value': 'F'}]
planes_agrupados = [
    'AP',   'Vida Entera',   'PALP',   'Temporal',
    'MetLife Pension',   'PUF',   'Variable Life',
    'Dotal',   'Life Cover Plus'
]
medios_pago = ['AUTOMATICO', 'MANUAL']


# ——————————————————————————————————————————————————————————
# 4. Layout de la aplicación
# ——————————————————————————————————————————————————————————
app.layout = html.Div([
    html.H1("Predicción de Cancelación"),
    html.P("Basado en características de la persona, predecir si va a cancelar o no el seguro"),

    html.Div([

        # Primera columna de campos de entrada en el dash
        html.Div([
            html.Label("Estado Inicial:"),
            dcc.Dropdown(id='Estado_Inicial', options=[{'label': '1', 'value': 1}, {'label': '0', 'value': 0}], value=0),

            html.Label("Huérfano:"),
            dcc.Dropdown(id='Huerfano', options=[{'label': '1', 'value': 1}, {'label': '0', 'value': 0}], value=0),

            html.Label("Días Mora:"),
            dcc.Input(id='dias_mora', type='number', value=0),

            html.Label("Antigüedad Póliza:"),
            dcc.Input(id='Ant_pol', type='number', value=0),
        ], style={'width': '23%', 'padding': '10px'}),

        # Segunda columna de campos de entrada en el dash
        html.Div([
            html.Label("NBS:"),
            dcc.Input(id='NBS', type='number', value=0),

            html.Label("Edad:"),
            dcc.Dropdown(id='Edad', options=[{'label': str(i), 'value': i} for i in range(18, 101)], value=30),

            html.Label("Total Mora:"),
            dcc.Dropdown(id='total_mora', options=[{'label': str(i), 'value': i} for i in range(0, 50)], value=0),

            html.Label("NBS Mora:"),
            dcc.Input(id='NBS_mora', type='number', value=0),
        ], style={'width': '23%', 'padding': '10px'}),

        # Tercera columna de campos de entrada en el dash
        html.Div([
            html.Label("Total Activas:"),
            dcc.Dropdown(id='Total_Activas', options=[{'label': str(i), 'value': i} for i in range(1, 50)], value=1),

            html.Label("NBS Vigente:"),
            dcc.Input(id='NBS_Vigente', type='number', value=0),

            html.Label("Plan Agrupado:"),
            dcc.Dropdown(id='Plan_Agrupado', options=[{'label': p, 'value': p} for p in planes_agrupados], value=planes_agrupados[0]),

            html.Label("Medio Pago:"),
            dcc.Dropdown(id='MedioPago', options=[{'label': m, 'value': m} for m in medios_pago], value=medios_pago[0]),
        ], style={'width': '23%', 'padding': '10px'}),

        # Cuarta columna de campos de entrada en el dash
        html.Div([
            html.Label("Género:"),
            dcc.Dropdown(id='genero', options=generos, value='M'),
        ], style={'width': '23%', 'padding': '10px'}),

    # Fin de las columnas de entrada        
    ], style={'display': 'flex', 'justify-content': 'space-between'}),

    # Botón para predecir
    html.Button('Predecir Cancelación', id='boton-prediccion', n_clicks=0),
    html.Div(id='resultado-prediccion', style={'font-size': '20px', 'margin-top': '20px'})

])


# ——————————————————————————————————————————————————————————
# 5. Callback: llamar a la API y mostrar lsa predicción
# ——————————————————————————————————————————————————————————
@app.callback(
    Output('resultado-prediccion', 'children'),
    Input('boton-prediccion', 'n_clicks'),
    State('Estado_Inicial', 'value'),
    State('Huerfano', 'value'),
    State('dias_mora', 'value'),
    State('Ant_pol', 'value'),
    State('NBS', 'value'),
    State('Edad', 'value'),
    State('total_mora', 'value'),
    State('NBS_mora', 'value'),
    State('Total_Activas', 'value'),
    State('NBS_Vigente', 'value'),
    State('Plan_Agrupado', 'value'),
    State('MedioPago', 'value'),
    State('genero', 'value'),
    prevent_initial_call=True
)

#"""  
#    [Input('Estado_Inicial', 'value'), Input('Huerfano', 'value'), Input('dias_mora', 'value'),
#    Input('Ant_pol', 'value'), Input('NBS', 'value'), Input('Edad', 'value'), Input('total_mora', 'value'),
#    Input('NBS_mora', 'value'), Input('Total_Activas', 'value'), Input('NBS_Vigente', 'value'),
#    Input('Plan_Agrupado', 'value'), Input('MedioPago', 'value'), Input('genero', 'value')],
#    prevent_initial_call=False)
#"""

#""" 
#    def predecir_cancelacion(Estado_Inicial, Huerfano, dias_mora, Ant_pol, NBS, Edad, total_mora, NBS_mora, Total_Activas, NBS_Vigente, Plan_Agrupado, MedioPago, genero):
#    entrada = pd.DataFrame({
#        'Estado_Inicial': [Estado_Inicial],
#        'Huerfano': [Huerfano],
#        'dias_mora': [dias_mora],
#        'Ant_pol': [Ant_pol],
#        'NBS': [NBS],
#        'Edad': [Edad],
#        'total_mora': [total_mora],
#        'NBS_mora': [NBS_mora],
#        'Total_Activas': [Total_Activas],
#        'NBS_Vigente': [NBS_Vigente],
#        'Plan_Agrupado': [Plan_Agrupado],
#        'MedioPago': [MedioPago],
#        'genero': [genero]
#    })
#"""

def predecir_cancelacion(n_clicks, Estado_Inicial, Huerfano, dias_mora, Ant_pol, NBS, Edad, total_mora, NBS_mora, Total_Activas, NBS_Vigente, Plan_Agrupado, MedioPago, genero):
    if not n_clicks:
        return html.Div("Presiona el botón para predecir la cancelación")

    payload = {
        "records": [
            {
                "Estado_Inicial": Estado_Inicial,
                "Huerfano": Huerfano,
                "dias_mora": dias_mora,
                "Ant_pol": Ant_pol,
                "NBS": NBS,
                "Edad": Edad,
                "total_mora": total_mora,
                "NBS_mora": NBS_mora,
                "Total_Activas": Total_Activas,
                "NBS_Vigente": NBS_Vigente,
                "Plan_Agrupado": Plan_Agrupado,
                "MedioPago": MedioPago,
                "genero": genero
            }
        ]
    }

    try:
        resp = requests.post(PREDICT_ENDPOINT, json=payload, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        pred = data["predictions"][0]
    except Exception as e:
        logger.error(f"Error llamando a la API: {e}")
        return html.Div(f"Error al llamar a la API: {e}", style={"color": "orange"})

    if pred == 1:
        return html.Div("El usuario va a cancelar", style={'color': 'red'})
    else:
        return html.Div("El usuario no va a cancelar", style={'color': 'black'})


# ——————————————————————————————————————————————————————————
# 6. Arranque de la aplicación
# ————————————————————————————————————————————————————————
if __name__ == '__main__':
#    app.run(debug=True)
    logger.info("Arrancando Dash sobre API en %s", PREDICT_ENDPOINT)
    app.run_server(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8050)),
        debug=True
    )



# %%



