# %%

#import sys
#!{sys.executable} -m pip install dash pandas --quiet

#import dash, pandas
#print("Dash v", dash.__version__)
#print("Pandas v", pandas.__version__)

#import sys
#!{sys.executable} -m pip install --quiet scikit-learn xgboost


# %%




import dash
from dash import dcc, html, Input, Output, State
import pickle
import pandas as pd

with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

# ── PARCHE PARA SOPORTAR DIFERENCIAS DE VERSIÓN ──
# En versiones modernas de XGBClassifier este atributo ya no existe,
# así que lo añadimos para que get_params() no explote:
setattr(model, "use_label_encoder", False)
# O también: model.use_label_encoder = False
# ────────────────────────────────────────────────

app = dash.Dash(__name__)
server = app.server

generos = [{'label': 'Masculino', 'value': 'M'}, {'label': 'Femenino', 'value': 'F'}]
planes_agrupados = [ 'AP',     'Vida Entera',            'PALP',        'Temporal',
 'MetLife Pension',             'PUF',   'Variable Life',           'Dotal',
 'Life Cover Plus']
medios_pago = ['AUTOMATICO', 'MANUAL']

app.layout = html.Div([
    html.H1("Predicción de Cancelación"),
    html.P("Basado en características de la persona, predecir si va a cancelar o no el seguro"),

    html.Div([
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

        html.Div([
            html.Label("Género:"),
            dcc.Dropdown(id='genero', options=generos, value='M'),
        ], style={'width': '23%', 'padding': '10px'}),
        
    ], style={'display': 'flex', 'justify-content': 'space-between'}),

    #html.Div(id='resultado-prediccion', style={'font-size': '20px', 'margin-top': '20px'})

    # Botón para predecir
    html.Button('Predecir Cancelación', id='boton-prediccion', n_clicks=0),
    html.Div(id='resultado-prediccion', style={'font-size': '20px', 'margin-top': '20px'})


])


@app.callback(
    Output('resultado-prediccion', 'children'),
    Input('boton-prediccion', 'n_clicks'),
    State('Estado_Inicial', 'value'),
    State('Huerfano', 'value'), State('dias_mora', 'value'),
    State('Ant_pol', 'value'), State('NBS', 'value'), State('Edad', 'value'), State('total_mora', 'value'),
    State('NBS_mora', 'value'), State('Total_Activas', 'value'), State('NBS_Vigente', 'value'),
    State('Plan_Agrupado', 'value'), State('MedioPago', 'value'), State('genero', 'value'),
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

    entrada = pd.DataFrame({
        'Estado_Inicial': [Estado_Inicial],
        'Huerfano': [Huerfano],
        'dias_mora': [dias_mora],
        'Ant_pol': [Ant_pol],
        'NBS': [NBS],
        'Edad': [Edad],
        'total_mora': [total_mora],
        'NBS_mora': [NBS_mora],
        'Total_Activas': [Total_Activas],
        'NBS_Vigente': [NBS_Vigente],
        'Plan_Agrupado': [Plan_Agrupado],
        'MedioPago': [MedioPago],
        'genero': [genero]
    })
    prediccion = model.predict(entrada)[0]

    if prediccion == 1:
        return html.Div("El usuario va a cancelar", style={'color': 'red'})
    else:
        return html.Div("El usuario no va a cancelar", style={'color': 'black'})


if __name__ == '__main__':
    app.run(debug=True)

# %%



