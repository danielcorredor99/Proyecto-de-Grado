import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, ClientsideFunction
from components.table.table import table
from components.table.table_info import TableInfo
import plotly.express as px

import numpy as np
import pandas as pd
from datetime import datetime as dt

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Volatilidad del cafe"

server = app.server
app.config.suppress_callback_exceptions = True

columnsPrediction=["Real","LSTM","RandomForest","XGBoost","ELM"]

"""
volatilidad
variables economicas
Índices de noticias
índices de la COVID-19

#Comparacion grafica de los resultados
#Resultados metricas de bondad de ajuste
"""
## order Durante:
""" Vol
Vol y VE
V e IN
V e IC
V, VE e IN
V, VE e IC
V, IN e IC
V, VE, IN e IC 


"""
best_models=pd.read_csv("data/unitModel/best_models.csv")
datasets2=[["Volatilidad","Volatilidad y Variables económicas","Volatilidad e índices de noticias","Volatilidad,Variables económicas e índices de noticias"],
          ["Volatilidad","Volatilidad y Variables económicas","Volatilidad e índices de noticias","Volatilidad e índices de la COVID-19",
           "Volatilidad,Variables económicas e índices de noticias","volatilidad ,variables e índices de la COVID-19","Volatilidad,índices de la COVID-19 e índices de noticias","Volatilidad,Variables económicas,índices de la COVID-19 e índices de noticias"]
        ]
datasets=[["Vol","Vol y VE","Vol e IN","Vol,VE e IN"],
          ["Vol","Vol y VE","Vol e IN","Vol e IC",
           "Vol,VE e IN","Vol ,VE e IC","Vol,IC e IN","Vol,VE,IC e IN"]
        ]

df_data=[[pd.read_csv("data/VolatilidadAntes.csv"),pd.read_csv("data/VolatilidadVariablesAntes.csv"),pd.read_csv("data/VolatilidadNoticiasAntes.csv")
         ,pd.read_csv("data/VolatilidadVariablesNoticiasAntes.csv")],
         [pd.read_csv("data/VolatilidadDespues.csv"),pd.read_csv("data/VolatilidadVariablesDespues.csv"),pd.read_csv("data/VolatilidadNoticiasDespues.csv"),
          pd.read_csv("data/VolatilidadIndicesDespues.csv"),pd.read_csv("data/VolatilidadVariablesNoticiasDespues.csv"),pd.read_csv("data/VolatilidadVariablesIndicesDespues.csv"),
          pd.read_csv("data/VolatilidadIndicesNoticiasDespues1.csv"),pd.read_csv("data/VolatilidadVariablesIndicesNoticiasDespues.csv")]]

models_name=["LSTM","RandomForest","XG-Boost","ELM"]
pfx_mdl2="data/unitModel"
type_models=["LSTM","Random Forest","XG-Boost","ELM"]
result_model_data=[[pd.read_csv(f"{pfx_mdl2}/LSTM test antes.csv",sep=";"),pd.read_csv(f"{pfx_mdl2}/RF test antes.csv",sep=";"),pd.read_csv(f"{pfx_mdl2}/XGB test antes.csv",sep=";")
         ,pd.read_csv(f"{pfx_mdl2}/ELM test antes.csv",sep=";")],
         [pd.read_csv(f"{pfx_mdl2}/LSTM test durante.csv",sep=";"),pd.read_csv(f"{pfx_mdl2}/RF test durante.csv",sep=";"),pd.read_csv(f"{pfx_mdl2}/XGB test durante.csv",sep=";")
         ,pd.read_csv(f"{pfx_mdl2}/ELM test durante.csv",sep=";")]
         ]

pfx_mdl="data/compareModels"
compare_model_data=[[pd.read_csv(f"{pfx_mdl}/Vol_antes.csv"),pd.read_csv(f"{pfx_mdl}/Vol y VE_antes.csv"),pd.read_csv(f"{pfx_mdl}/Vol e IN_antes.csv")
         ,pd.read_csv(f"{pfx_mdl}/Vol, VE e IN_antes.csv")],
         [pd.read_csv(f"{pfx_mdl}/Vol_durante.csv"),pd.read_csv(f"{pfx_mdl}/Vol y VE_durante.csv"),pd.read_csv(f"{pfx_mdl}/Vol e IN_durante.csv"),
          pd.read_csv(f"{pfx_mdl}/Vol e IC_durante.csv"),pd.read_csv(f"{pfx_mdl}/Vol, VE e IN_durante.csv"),pd.read_csv(f"{pfx_mdl}/Vol, VE e IC_durante.csv"),
          pd.read_csv(f"{pfx_mdl}/Vol, IN e IC_durante.csv"),pd.read_csv(f"{pfx_mdl}/Vol, VE, IN e IC_durante.csv")]]

table_data=[compare_model_data,result_model_data]

params1 = {
            'title': 'Comparacion', 
            'description': 'last measured data',
            'columns':list(compare_model_data[0][0].columns)
}
params2 = {
            'title': f'Resultados del modelo Random Forest', 
            'description': '',
            'columns':list(result_model_data[0][0].columns)
}
params3 = {
            'title': f'Mejores Modelos', 
            'description': '',
            'columns':list(best_models.columns)
}
tb=table(compare_model_data[1][0],params1)
tb2=table(result_model_data[1][0],params2)

tb_info=TableInfo(best_models,params3)
def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H3("Modelos de pronóstico para la volatilidad de los futuros del café colombiano basado en noticias antes y durante la pandemia de la COVID-19 y variables económicas."),
            html.Div(
                id="intro",
                children="Este Dashboard presenta los resultados de las predicciones antes y durante la pandemia de la COVID-19de los modelos Random Forest (RF), Extreme Gradient Boosting (XGB), Extreme Learning Machine (ELM) y Long Short-Term Memory (LSTM), junto con las variables de entrada volatilidad de los futuros del café colombiano (Vol), Variables Económicas (VE), Índices de Noticias (IN) e Índices de noticias de la COVID-19 (IC)."
            ),
           
        ],
    )


def control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.Hr(),
            html.H5("Comparación gráfica de los resultados"),
            
            html.P("Seleccione momento del estudio"),
            dcc.Dropdown(
                id="moment-select",
                options=[{"label": "Antes de pandemia", "value": 0},{"label": "Durante pandemia", "value": 1}],
                value=0,
            ),
            
            html.Br(),
            html.P("Seleccione dataset"),
            dcc.Dropdown(
                id="dataset-select",
                multi=False
            ),
            html.Br(),
            html.P("Filtrar por tipo de dato"),
            dcc.Dropdown(
                id="filter_type_data",
                options=[{"label": "Todo", "value": 0},{"label": "Entrenamiento", "value": 1},{"label": "Test", "value": 2}],
                value=0,
                multi=False
            ),
            
            html.Br(),
            html.Hr(),
            html.H5("Resultados métricas de bondad de ajuste"),
            
            #Control table
            html.Br(),
            html.P("Seleccione tipo de resultado"),
            dcc.Dropdown(
                id="filter_table_type_data",
                options=[{"label": "Comparación de modelos", "value": 0},{"label": "Rendimiento individual de un modelo", "value": 1}],
                value=0,
                multi=False
            ),
            html.Br(),
            html.P("Seleccione dataSet",id="filter_table_name"),
            dcc.Dropdown(
                id="dataset_table_select",
                multi=False
            ),
            html.Br(),
            html.P("Seleccione un modelo",id="model_name"),
            dcc.Dropdown(
                id="model-select-table",
                options=[{"label": "LSTM", "value": 0},{"label": "RandomForest", "value": 1},
                         {"label": "XG-Boost", "value": 2},{"label": "ELM", "value": 3}],
                value=0,
            ),
            html.Br(),
            html.P("Seleccione momento del estudio",id="moment_name"),
            dcc.Dropdown(
                id="moment-select-table",
                options=[{"label": "Antes de pandemia", "value": 0},{"label": "Durante  pandemia", "value": 1}],
                value=0,
            ),
            
            
            
        ],
    )

### Vol, VE, IN e IC
app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src="/assets/uis2.png")],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), control_card()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),
        
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
            children=[
                html.Br(),
                
                html.Div(
                    id="prediction_graph",
                    style={"margin-top":"30px","margin-bottom":"30px"},
                    children=[
                        html.Hr(),
                        dcc.Graph(id="result_prediction"),
                    ],
                ),
                 html.Div(
                    id="wait_time_card",
                    children=[
                        tb.display()],
                ),
                html.Div(
                    id="info_table2",
                    children=[
                        tb_info.display()],
                ),
                
            ],
        ),
    ],
)


@app.callback(
    Output("result_prediction", "figure"),
    [Input("dataset-select", "value"),Input("moment-select", "value"),Input("filter_type_data","value")],
)
def dataset_option(dataset_select,moment_select,filter_type_data):
    df=df_data[moment_select][dataset_select]
    
    if filter_type_data!=0:
        df=df[df["typeXGBoost"]=="train"] if filter_type_data==1 else df[df["typeXGBoost"]=="test"]
    model_name=datasets[moment_select][dataset_select]
    moment_name="Antes de pandemia" if moment_select==0 else "Durante pandemia"
    df.rename(columns={"Vol. Futuros":"Real"},inplace=True)
    fig1 = px.line(df, x="Fecha", y=columnsPrediction,
                title=f'<b style="font-size: 20px;">Resultados predicción con el dataset {model_name} </b>'+
                '<br>'+
                f'<b style="font-size: 15px;">{moment_name} </b>',
                labels={"value":"Volatilidad","variable":"Modelo"}
                )
    return fig1

@app.callback(
    Output("dataset-select", "options"),
    Output("dataset-select", "value"),
    [Input("moment-select", "value")],
)
def dataset_option(value_dataset):
    if value_dataset==0:
        options=[{"label":datasets[0][i],"value":i} for i in range(4)]
        return options,0
    options=[{"label":datasets[1][i],"value":i} for i in range(8)]
    return options,0

@app.callback(
    Output("dataset_table_select", "options"),
    Output("dataset_table_select", "value"),
    Output("moment-select-table", "style"),
    Output("moment_name", "style"),
    Output("filter_table_name", "style"),
    Output("dataset_table_select", "style"),
    Output("model_name", "style"),
    Output("model-select-table", "style"),
    [Input("filter_table_type_data", "value")],
)
def dataset_table_option(filter_table_type_data):
    
    if filter_table_type_data==0:
        options=[{"label":datasets[0][i]+" AP","value":i} for i in range(4)]
        options.extend([{"label":datasets[1][i]+" DP","value":i+4} for i in range(8)])

        return options,0,{"display":"none"},{"display":"none"},{},{},{"display":"none"},{"display":"none"}
    else:
        options=[{"label":type_models[i],"value":i} for i in range(4)]
        return options,0,{},{},{"display":"none"},{"display":"none"},{},{}
    
@app.callback(
        Output("wait_time_card","children"),
    [Input("filter_table_type_data", "value"),
    Input("dataset_table_select", "value"),
    Input("model-select-table", "value"),
    Input("moment-select-table", "value"),],
)
def set_table_option(type_table,df_select,model_select,moment_select):
    
    if type_table==0:
        dif=0 if df_select-4<0 else 1
        moment_name="Antes de pandemia" if dif==0 else "Durante pandemia"
        name_dataset=datasets2[dif][df_select] if dif<0 else datasets2[dif][df_select-4] 
        df=table_data[0][0][df_select] if dif<0 else table_data[0][1][df_select-4]
        params = {
                'title': f'Comparacion rendimiento de los modelos para el dataset {name_dataset}', 
                'description': f'Periodo: {moment_name}',
                'columns':list(df.columns)
                }
        tb.set_data(df,params)
        return tb.display()
    else:
        name_model=models_name[model_select]
        moment_name="antes de pandemia" if moment_select==0 else "durante pandemia"
        df=table_data[1][moment_select][model_select]
        params = {
                'title': f'Rendimiento del Modelo {name_model} {moment_name} ', 
                'description': 'last measured data',
                'columns':list(df.columns)
                }
        tb.set_data(df,params)
        return tb.display()


# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)