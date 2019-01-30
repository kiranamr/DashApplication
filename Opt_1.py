
import pandas as pd
import numpy as np

from pandas import DataFrame

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from textwrap import dedent as d
import json
from pandas.io.json import json_normalize

from dash.dependencies import Input, Output, State

import flask


url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=True),
    html.Div(id='page-content')
])

index=0


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

MainDataF = pd.read_excel('/home/dst/Documents/Data_Science/all_reports.xlsx', sheet_name='AnalysisResults', index_col=None)
data_xls = MainDataF.copy()
data_xls = data_xls.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
q2 = data_xls.quantile(.25, axis=0)

q4 = data_xls.quantile(.75, axis=0)
print(MainDataF.shape)

dff=DataFrame()

class read:

    def reder(a):
        global MainDataF
        global data_xls

        #data_xls=MainDataF.copy()#pd.read_excel('/home/dst/Documents/Data_Science/all_reports.xlsx', sheet_name='AnalysisResults', index_col=None)
        b = data_xls.shape
        '''
        X_Matrix = [[0 for x in range(4)] for y in range(b[0])]
        for row in range(b[0]):
            X_Matrix[row][0]= data_xls.iloc[row][0]
            '''
        MainData=MainDataF.copy()
        #data_xls=data_xls.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)

        #q2 = data_xls.quantile(.25, axis=0)

        #q4 = data_xls.quantile(.75, axis=0)

        print("75 % of dencity ",q4[1])

        print("25 % of dencity ", q2[1])
        ls=list(data_xls.columns.values)
        print(ls.__len__())
        print(q2[1])




        mainDataFrame=data_xls.copy()
        OutOfRange=data_xls.copy()
        BelowRange=data_xls.copy()

        for index in range(ls.__len__()):
            arr = np.array([])
            l=ls[index]

            mainDataFrame.loc[mainDataFrame[l] < q2[index], l] = 0
            mainDataFrame.loc[mainDataFrame[l] > q4[index], l] = 0
            OutOfRange.loc[OutOfRange[l] < q4[index], l] = 0
            BelowRange.loc[BelowRange[l] > q2[index], l] = 0

        print('in renge',mainDataFrame)
        print('out of range',OutOfRange)
        print('below range',BelowRange)



        InRengeArray=np.count_nonzero(mainDataFrame, axis=1)
        OutOfRangeArray=np.count_nonzero(OutOfRange, axis=1)
        BelowRangeArray=np.count_nonzero(BelowRange, axis=1)

        print('Inrenge',InRengeArray)
        print('Outof', OutOfRangeArray)
        print('Below', BelowRangeArray)

        mainDataFrame['IQR_mean']=mainDataFrame.mean(axis=1)
        OutOfRange['IQR_mean'] = OutOfRange.mean(axis=1)
        BelowRange['IQR_mean'] = BelowRange.mean(axis=1)

        RangeDataFrame = DataFrame()
        RangeDataFrame['inrange'] = pd.Series(InRengeArray)
        RangeDataFrame['outrange'] = pd.Series(OutOfRangeArray)
        RangeDataFrame['below'] = pd.Series(BelowRangeArray)


        RangeDataFrame['Range'] = RangeDataFrame.idxmax(axis=1)

        arr1=np.array([])
        for row in range(b[0]):
            if RangeDataFrame.iloc[row][3]=='inrange':
                arr1=np.append(arr1,mainDataFrame.iloc[row][ls.__len__()])
            elif RangeDataFrame.iloc[row][3]=='below':
                arr1=np.append(arr1,BelowRange.iloc[row][ls.__len__()])
            else: arr1=np.append(arr1,OutOfRange.iloc[row][ls.__len__()])
        print(RangeDataFrame)
        MainData['Row_Main']=MainData.mean(axis=1)
        df=DataFrame()
        df['SerialNames']=pd.Series(MainData['SampleNumber'])

        df['Row_Main']=pd.Series(MainData['Row_Main'])
        df['IQR_main'] = pd.Series(arr1)
        df['Range'] = pd.Series(RangeDataFrame['Range'])



        print(df)

        return df
    def AccessRow(index):
        global MainDataF
        global dff

        data = MainDataF.copy()
        b = data.shape
        data = data.apply(lambda x: pd.to_numeric(x, errors='coerce')).fillna(0)
        q2 = data.quantile(.25, axis=0)

        q4 = data.quantile(.75, axis=0)

        row=data.iloc[index]


        ar=np.array([])
        for l in range(b[1]):
            ar=np.append(ar,data.iloc[index][l])
        print(ar)


        dataFrame = pd.DataFrame()
        dataFrame['Header']=pd.Series(list(MainDataF.columns.values))
        dataFrame['Row_Data']=pd.Series(ar)
        bellow=np.where(np.array(row)<np.array(q2))
        outter=np.where(np.array(row)>np.array(q4))
        inRange=np.where(np.logical_and(np.array(row)>np.array(q2), np.array(row)<np.array(q4)))
        arr=np.array([])
        for clm in range(inRange.__len__()):
            dataFrame.loc[inRange[clm], 'Range']='In_Range'
        for clm in range(bellow.__len__()):
            dataFrame.loc[bellow[clm], 'Range']='Below_Range'
        for clm in range(outter.__len__()):
            dataFrame.loc[outter[clm], 'Range']='Out_Range'
        #dataFrame.index.name='index'
        m=np.array([])
        #print('printing row',row)
        for r in range(0, 180):
            m=np.append(m, r)
        dataFrame['Index']=pd.Series(m)
        dff= dataFrame.copy()
        return dff

df= read.reder(1)

print(dff)
dff= read.AccessRow(index)

layout_1 = html.Div([

    dcc.Link(
        dcc.Graph(
            id='life-exp-vs-gdp',
            figure={
                'data': [
                    go.Scatter(
                        x=df[df['Range'] == i]['IQR_main'],
                        y=df[df['Range'] == i]['Row_Main'],
                        text=df[df['Range'] == i]['SerialNames'],
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                        name=i
                    ) for i in df.Range.unique()
                ],
                'layout': {
                    'clickmode': 'event+select'
                },
                'layout': go.Layout(
                    xaxis={'type': 'log', 'title': 'IQR Mean'},
                    yaxis={'title': 'Average'},
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                    legend={'x': 0, 'y': 1},
                    hovermode='closest'

                )
            }
        ),
        href='/page-1'),
        html.Pre(id='click-data', style=styles['pre']),




])



@app.callback(
    Output('click-data', 'children'),
    [Input('life-exp-vs-gdp', 'clickData')])
def display_click_data(clickData):
    global dff,index
    dff1= json.dumps(clickData, indent=2)
    resp = json.loads(dff1)
    index=resp["points"][0]["pointNumber"]
    #print(resp)

    dff = read.AccessRow(index).copy()
    return dff.to_json(date_format='iso', orient='split')



@app.callback(
    dash.dependencies.Output('page-content', 'children'),
    [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    global index, dff
    print('in link call back fun', index)
    print('in link call back fun', dff)

    if pathname == '/page-1':
        return layout_2
    else:
        return layout_1

layout_2 =html.Div([

    dcc.Graph(
        id='life-exp-vs-gdp',
        figure={
            'data': [
                go.Scatter(
                    x=dff[dff['Range'] == i]['Index'],
                    y=dff[dff['Range'] == i]['Row_Data'],
                    text=dff[dff['Range'] == i]['Header'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=i
                ) for i in dff.Range.unique()
            ],

            'layout': go.Layout(
                xaxis={'type': 'log', 'title': 'IQR Mean'},
                yaxis={'title': 'Average'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    ),

])


if __name__ == '__main__':
    app.run_server(debug=True)