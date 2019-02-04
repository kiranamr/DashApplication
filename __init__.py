import base64
import os
from urllib.parse import quote as urlquote

from flask import Flask, send_from_directory
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import State, Input, Output

import pandas as pd
from pandas import DataFrame
import numpy as np
import json
import plotly.graph_objs as go
import shutil

dff = DataFrame()
MainDataF = DataFrame()

UPLOAD_DIRECTORY = "/var/www/html/FlaskApp/FlaskApp/Data_Science/"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(server=server)


@server.route("/download/<path:path>")
def download(path):
    """Serve a file from the upload directory."""
    return send_from_directory(UPLOAD_DIRECTORY, path, as_attachment=True)
data=[]

app.layout = html.Div([
    html.Div([
        html.H1("File Browser"),
        html.H2("Upload"),
        dcc.Upload(
            id="upload-data",
            children=html.Div(
                html.A('Select Files')
            ),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
            multiple=True,
        ),
        html.Button('Update Graph',n_clicks=0, id='submit-button'),
    ],
    id='Content_style',
    style = {"max-width": "500px",
             'display':'block'},

    ),

    html.Div([
        #html.H2("File List"),
        #html.Ul(id="file-list"),
        dcc.Graph(
            id='life-exp-vs-gdp',

                figure={

                    'data': data,
                    'layout': {
                        'clickmode': 'event+select'
                    },
                    'layout': go.Layout(
                        xaxis={'type': 'log', 'title': 'IQR Mean'},
                        yaxis={'type': 'log', 'title': 'Average'},
                        margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                        legend={'x': 0, 'y': 1},
                        hovermode='closest'
                    ),
                }
            )

        ],
        id='graph_styles',

        )
    ])


def save_file(name, content):
    """Decode and store a file uploaded with Plotly Dash."""
    data = content.encode("utf8").split(b";base64,")[1]
    if not os.path.isdir(UPLOAD_DIRECTORY):
        os.mkdir(UPLOAD_DIRECTORY)

    #for file in request.files.getlist("file"):
    #destination = shutil.move(os.path.join(UPLOAD_DIRECTORY, 'all_reports.xlsx'), os.path.join(UPLOAD_DIRECTORY, 'all_reports.xlsx'))
    #fp.write(base64.decodebytes(data))
    with open(os.path.join(UPLOAD_DIRECTORY, 'all_reports.xlsx'), "wb") as fp:
        fp.write(base64.decodebytes(data))


def uploaded_files():
    """List the files in the upload directory."""
    files = []
    for filename in os.listdir(UPLOAD_DIRECTORY):
        path = os.path.join(UPLOAD_DIRECTORY, filename)
        if os.path.isfile(path):
            files.append(filename)
    return files


def file_download_link(filename):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    location = "/download/{}".format(urlquote(filename))
    return html.A(filename, href=location)

def LoadFile():
    global MainDataF
    MainDataF = pd.read_excel('/var/www/html/FlaskApp/FlaskApp/Data_Science//all_reports.xlsx',
                              sheet_name='AnalysisResults',
                              index_col=None)
    data=readData()
    return data


def readData():
    global MainDataF
    #global data
    data_xls = MainDataF.copy()
    data_xls = data_xls.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    q2 = data_xls.quantile(.25, axis=0)

    q4 = data_xls.quantile(.75, axis=0)
    print(MainDataF.shape)

    b = data_xls.shape
    MainData = MainDataF.copy()
    print("75 % of dencity ", q4[1])

    print("25 % of dencity ", q2[1])
    ls = list(data_xls.columns.values)
    print(ls.__len__())
    print(q2[1])

    mainDataFrame = data_xls.copy()
    OutOfRange = data_xls.copy()
    BelowRange = data_xls.copy()

    for index in range(ls.__len__()):
        arr = np.array([])
        l = ls[index]

        mainDataFrame.loc[mainDataFrame[l] < q2[index], l] = 0
        mainDataFrame.loc[mainDataFrame[l] > q4[index], l] = 0
        OutOfRange.loc[OutOfRange[l] < q4[index], l] = 0
        BelowRange.loc[BelowRange[l] > q2[index], l] = 0
    print('in renge', mainDataFrame)
    print('out of range', OutOfRange)
    print('below range', BelowRange)

    InRengeArray = np.count_nonzero(mainDataFrame, axis=1)
    OutOfRangeArray = np.count_nonzero(OutOfRange, axis=1)
    BelowRangeArray = np.count_nonzero(BelowRange, axis=1)

    print('Inrenge', InRengeArray)
    print('Outof', OutOfRangeArray)
    print('Below', BelowRangeArray)

    mainDataFrame['IQR_mean'] = mainDataFrame.mean(axis=1)
    OutOfRange['IQR_mean'] = OutOfRange.mean(axis=1)
    BelowRange['IQR_mean'] = BelowRange.mean(axis=1)

    RangeDataFrame = DataFrame()
    RangeDataFrame['In_Range'] = pd.Series(InRengeArray)
    RangeDataFrame['Outof_Range'] = pd.Series(OutOfRangeArray)
    RangeDataFrame['Below_Range'] = pd.Series(BelowRangeArray)

    RangeDataFrame['Range'] = RangeDataFrame.idxmax(axis=1)

    arr1 = np.array([])
    for row in range(b[0]):
        if RangeDataFrame.iloc[row][3] == 'In_Range':
            arr1 = np.append(arr1, mainDataFrame.iloc[row][ls.__len__()])
        elif RangeDataFrame.iloc[row][3] == 'Below_Range':
            arr1 = np.append(arr1, BelowRange.iloc[row][ls.__len__()])
        else:
            arr1 = np.append(arr1, OutOfRange.iloc[row][ls.__len__()])
    print(RangeDataFrame)
    MainData['Row_Main'] = MainData.mean(axis=1)
    df = DataFrame()
    df['SerialNames'] = pd.Series(MainData['SampleNumber'])
    df['Row_Main'] = pd.Series(MainData['Row_Main'])
    df['IQR_main'] = pd.Series(arr1)
    df['Range'] = pd.Series(RangeDataFrame['Range'])
    print(df)
    for i in df.Range.unique():
        if i == 'In_Range':
            trace0 = go.Scatter(
                x=df[df['Range'] == i]['IQR_main'],
                y=df[df['Range'] == i]['Row_Main'],
                text=df[df['Range'] == i]['SerialNames'],
                mode='markers',
                opacity=0.7,
                name='In_Range',
                marker=dict(
                    size=15,
                    color='#3bbd0d',
                    line=dict(
                        width=0.5,
                        color='#ffffff'
                    )

                )
            )
        elif i == 'Outof_Range':
            trace1 = go.Scatter(
                x=df[df['Range'] == i]['IQR_main'],
                y=df[df['Range'] == i]['Row_Main'],
                text=df[df['Range'] == i]['SerialNames'],
                name='Outof_Range',
                mode='markers',
                marker=dict(
                    size=15,
                    color=' #208ea4 ',
                    line=dict(
                        width=0.5,
                        color='#ffffff'
                    )

                )
            )
        else:
            trace2 = go.Scatter(
                x=df[df['Range'] == i]['IQR_main'],
                y=df[df['Range'] == i]['Row_Main'],
                text=df[df['Range'] == i]['SerialNames'],
                name='Below_Range',
                mode='markers',
                marker=dict(
                    size=15,
                    color='#f06206',
                    line=dict(
                        width=0.5,
                        color='#ffffff'
                    )

                )
            )

    data = [trace0, trace1, trace2]
    return data

def AccessRow(index):
    global MainDataF
    global dff

    data = MainDataF.copy()
    b = data.shape
    data = data.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
    q2 = data.quantile(.25, axis=0)

    q4 = data.quantile(.75, axis=0)

    row = data.iloc[index]

    ar = np.array([])
    for l in range(b[1]):
        ar = np.append(ar, data.iloc[index][l])
    print(ar)

    dataFrame = pd.DataFrame()
    dataFrame['Header'] = pd.Series(list(MainDataF.columns.values))
    dataFrame['Row_Data'] = pd.Series(ar)
    bellow = np.where(np.array(row) < np.array(q2))
    outter = np.where(np.array(row) > np.array(q4))
    inRange = np.where(np.logical_and(np.array(row) > np.array(q2), np.array(row) < np.array(q4)))
    arr = np.array([])
    for clm in range(inRange.__len__()):
        dataFrame.loc[inRange[clm], 'Range'] = 'In_Range'
    for clm in range(bellow.__len__()):
        dataFrame.loc[bellow[clm], 'Range'] = 'Below_Range'
    for clm in range(outter.__len__()):
        dataFrame.loc[outter[clm], 'Range'] = 'Outof_Range'
    # dataFrame.index.name='index'
    m = np.array([])
    # print('printing row',row)
    for r in range(0, 180):
        m = np.append(m, r)
    dataFrame['Index'] = pd.Series(m)
    dff = dataFrame.copy()
    return dff


@app.callback(Output("graph_styles", "style"),
              [Input('submit-button','value')],

              )
def update_styles(n_click):
    print("printing n_click ---------" ,n_click )
    return {style : {"max-width": "500px",
                     "display": 'none'}
    }
    #if n_click>0:
        #n_click={"max-width": "500px", "display": 'block'}

@app.callback(
    Output("life-exp-vs-gdp", "figure"),
    [Input("upload-data", "filename"), Input("upload-data", "contents"),
     Input('life-exp-vs-gdp', 'clickData')],
)
def update_output(uploaded_filenames, uploaded_file_contents, clickData):
    """Save uploaded files and regenerate the file list."""
    print("--------------------",clickData)
    print("file name ",uploaded_filenames)
    print("file contains ", uploaded_filenames)

    if clickData is None:
        if uploaded_filenames is not None and uploaded_file_contents is not None:
            for name, data in zip(uploaded_filenames, uploaded_file_contents):
                save_file(name, data)

        files = uploaded_files()
        data=LoadFile()

        #if len(files) == 0:
         #   return [html.Li("No files yet!")]
        #else:
         #   return [html.Li(file_download_link(filename)) for filename in files]
        return {'data': data,
         'layout': go.Layout(
             xaxis={'type': 'log', 'title': 'IQR Mean'},
             yaxis={'type': 'log', 'title': 'Average'},
             margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
             legend={'x': 0, 'y': 1},
             hovermode='closest'
         )
         }
    else:
        print('PRINTING CLICKED DATA', clickData)
        global dff, index
        dff1 = json.dumps(clickData, indent=2)
        resp = json.loads(dff1)
        index = resp["points"][0]["pointNumber"]
        # print(resp)

        dff = AccessRow(index).copy()
        print(dff)
        for i in dff.Range.unique():
            if i == 'In_Range':
                trace0 = go.Scatter(
                    x=dff[dff['Range'] == i]['Index'],
                    y=dff[dff['Range'] == i]['Row_Data'],
                    text=dff[dff['Range'] == i]['Header'],
                    mode='markers',
                    opacity=0.7,
                    name='In_Range',
                    marker=dict(
                        size=15,
                        color=' #3bbd0d',
                        line=dict(
                            width=0.5,
                            color='#ffffff'
                        )

                    )
                )
            elif i == 'Outof_Range':
                trace1 = go.Scatter(
                    x=dff[dff['Range'] == i]['Index'],
                    y=dff[dff['Range'] == i]['Row_Data'],
                    text=dff[dff['Range'] == i]['Header'],
                    name='Outof_Range',
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=' #208ea4 ',
                        line=dict(
                            width=0.5,
                            color='#ffffff'
                        )

                    )
                )
            else:
                trace2 = go.Scatter(
                    x=dff[dff['Range'] == i]['Index'],
                    y=dff[dff['Range'] == i]['Row_Data'],
                    text=dff[dff['Range'] == i]['Header'],
                    name='Below_Range',
                    mode='markers',
                    marker=dict(
                        size=15,
                        color='#f06206',
                        line=dict(
                            width=0.5,
                            color='#ffffff'
                        )

                    )
                )

        data = [trace0, trace1, trace2]
        return {'data': data,
                'layout': go.Layout(
                    xaxis={'type': 'log', 'title': 'IQR Mean'},
                    yaxis={'type': 'log', 'title': 'Average'},
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                    legend={'x': 0, 'y': 1},
                    hovermode='closest'
                )
                }



if __name__ == "__main__":
    app.run_server(debug=True, port=8888)
