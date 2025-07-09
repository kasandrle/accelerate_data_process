import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
from os import listdir
from os.path import isfile, join

# Load and parse data
path = './dummydata/outgassing_data/'
onlyfiles_outgassing = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('_outgassing_data_mean_std.txt')]
onlyfiles_TEY = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('_TEY_normalized.txt')]

def load_data(files, suffix):
    data = []
    for file in files:
        label = file.replace(suffix, '')
        df = pd.read_csv(join(path, file), sep='\t', skiprows=1)
        df['name'] = label
        data.append(df)
    return pd.concat(data, ignore_index=True)

df_outgassing = load_data(onlyfiles_outgassing, '_outgassing_data_mean_std.txt')
df_TEY = load_data(onlyfiles_TEY, '_TEY_normalized.txt')

# Start Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("Outgassing Spectrum Viewer"),
    dcc.Dropdown(
        id='sample-dropdown',
        options=[{'label': name, 'value': name} for name in sorted(df_outgassing['name'].unique())],
        value=[df_outgassing['name'].unique()[0]],  # Now a list
        multi=True
    ),
    dcc.Graph(id='outgassing-plot'),
    html.H2("TEY-Shutter Data"),
    dcc.Dropdown(
        id='sample-dropdown_TEY',
        options=[{'label': name, 'value': name} for name in sorted(df_TEY['name'].unique())],
        value=[df_TEY['name'].unique()[0]],
        multi=True
    ),
    dcc.Graph(id='tey-plot')  # New graph
])


@app.callback(
    Output('outgassing-plot', 'figure'),
    Input('sample-dropdown', 'value')
)
def update_figure(selected_samples):
    fig = go.Figure()

    if not selected_samples:
        return fig

    for sample in selected_samples:
        sub_df = df_outgassing[df_outgassing['name'] == sample]
        avg = sub_df['Avg Values (Torr)'].clip(lower=0)
        std = sub_df['Std Values (Torr)']
        std = std.where(std <= avg, 0)

        fig.add_trace(go.Bar(
            x=list(range(1, len(avg) + 1)),
            y=avg,
            error_y=dict(type='data', array=std, visible=True),
            name=sample,
            opacity=0.6
        ))

    fig.update_layout(
        title="Outgassing Spectrum",
        xaxis_title="m/z",
        yaxis_title="Ion signal (Torr)",
        template="plotly_white",
        barmode='group'
    )
    return fig

@app.callback(
    Output('tey-plot', 'figure'),
    Input('sample-dropdown_TEY', 'value')
)
def update_tey_plot(selected_samples):
    fig = go.Figure()

    if not selected_samples:
        return fig

    for sample in selected_samples:
        sub_df = df_TEY[df_TEY['name'] == sample]
        fig.add_trace(go.Scatter(
            x=sub_df['Time(s)'],
            y=sub_df['Normalized_TEY'],
            mode='lines',
            name=sample
        ))

    fig.update_layout(
        title="TEY-Shutter Data",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (A)",
        template="plotly_white"
    )
    return fig

# Run app
if __name__ == "__main__":
    app.run_server(debug=True, port=8080)