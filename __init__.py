
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import json
import graph

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash()#__name__, external_stylesheets=external_stylesheets)

#app.config.suppress_callback_exceptions = True


df = graph.read.reder()
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

app.layout = html.Div([
    # dcc.Link(
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
])


@app.callback(
    Output('life-exp-vs-gdp', 'figure'),
    [Input('life-exp-vs-gdp', 'clickData')])
def display_click_data(clickData):
    print('PRINTING CLICKED DATA', clickData)
    global dff, index
    dff1 = json.dumps(clickData, indent=2)
    resp = json.loads(dff1)
    index = resp["points"][0]["pointNumber"]
    # print(resp)

    dff = graph.read.AccessRow(index).copy()
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

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)

