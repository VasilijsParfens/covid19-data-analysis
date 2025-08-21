# visualization/app.py

import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import statsmodels.api as sm
import requests
import os

# ==================================
# Configuration
# ==================================
API_BASE = os.environ.get("API_BASE", "http://localhost:8000")  # URL for FastAPI backend
DATA_FILE = os.environ.get("DATA_FILE", "data/enriched_covid_data_filtered.csv")
FIG_TEMPLATE = os.environ.get("PLOTLY_TEMPLATE", "plotly_dark")  # "plotly_white" or "plotly_dark"

# Prophet can be heavy; guard import so the rest of the app still runs
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# ==================================
# Load initial dataset
# ==================================
df = pd.read_csv(DATA_FILE)
countries = sorted(df['COUNTRY_REGION'].dropna().unique())

# ==================================
# Helper Functions
# ==================================
def fetch_historical_data(country: str) -> pd.DataFrame:
    """Fetch historical time series (cases) for a country from backend."""
    try:
        resp = requests.get(f"{API_BASE}/historical-data/", params={"country": country.lower()})
        resp.raise_for_status()
        historical_data = resp.json().get('historical_data', [])
        if not isinstance(historical_data, list) or not historical_data:
            return pd.DataFrame()
        df_hist = pd.DataFrame(historical_data)
        if 'DATE' not in df_hist.columns:
            return pd.DataFrame()
        df_hist['DATE'] = pd.to_datetime(df_hist['DATE'], errors='coerce')
        df_hist.dropna(subset=['DATE'], inplace=True)
        # Expecting cases column as 'CASES'; rename for Prophet
        if 'CASES' not in df_hist.columns:
            return pd.DataFrame()
        df_hist.rename(columns={'DATE': 'ds', 'CASES': 'y'}, inplace=True)
        # ensure numeric
        df_hist['y'] = pd.to_numeric(df_hist['y'], errors='coerce').fillna(0)
        return df_hist[['ds', 'y']].sort_values('ds')
    except Exception as e:
        print(f"Error fetching historical data for {country}: {e}")
        return pd.DataFrame()

def make_forecast(df_hist: pd.DataFrame, horizon_days: int = 30) -> pd.DataFrame:
    """Create Prophet forecast for given horizon."""
    if not PROPHET_AVAILABLE:
        return go.Figure(layout=dict(title="Forecast unavailable — Prophet not installed"))
    if df_hist.empty or not {'ds', 'y'}.issubset(df_hist.columns):
        return pd.DataFrame()
    try:
        model = Prophet(daily_seasonality=True)
        model.fit(df_hist)
        future = model.make_future_dataframe(periods=horizon_days)
        forecast = model.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        print(f"Forecast generation failed: {e}")
        return pd.DataFrame()

def fit_population_density_model(filtered_df: pd.DataFrame) -> pd.DataFrame:
    """Fit log-log OLS: Cases_per_Million ~ Population_Density_Calc."""
    if filtered_df.empty:
        filtered_df = filtered_df.copy()
        filtered_df['fitted'] = pd.Series(dtype=float)
        return filtered_df
    try:
        temp = filtered_df[['Population_Density_Calc', 'Cases_per_Million']].replace([np.inf, -np.inf], np.nan).dropna()
        if temp.empty:
            filtered_df = filtered_df.copy()
            filtered_df['fitted'] = pd.Series(dtype=float)
            return filtered_df

        X = np.log(temp['Population_Density_Calc'] + 1)
        Y = np.log(temp['Cases_per_Million'] + 1)
        X_sm = sm.add_constant(X)
        model = sm.OLS(Y, X_sm).fit()

        # Predict on all rows where Population_Density_Calc exists
        filtered_df = filtered_df.copy()
        X_all = np.log(filtered_df['Population_Density_Calc'].fillna(0) + 1)
        X_all_sm = sm.add_constant(X_all, has_constant='add')
        preds = model.predict(X_all_sm)
        filtered_df['fitted'] = np.exp(preds) - 1
    except Exception as e:
        print(f"Error fitting population density model: {e}")
        filtered_df = filtered_df.copy()
        filtered_df['fitted'] = pd.Series(dtype=float)
    return filtered_df

def fetch_clusters(k=3) -> pd.DataFrame:
    """Fetch precomputed clusters from backend."""
    try:
        resp = requests.get(f"{API_BASE}/cluster-countries/", params={"k": int(k)})
        resp.raise_for_status()
        data = resp.json()
        return pd.DataFrame(data.get("clusters", []))  # expects columns: COUNTRY_REGION, cluster
    except Exception as e:
        print(f"Error fetching clusters: {e}")
        return pd.DataFrame()

# ==================================
# Figure Builders
# ==================================
def build_cases_vs_deaths_figure(filtered_df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        filtered_df,
        x="Cases_per_Million",
        y="Deaths_per_Million",
        size="Population_2020",
        color="CFR_%",
        hover_name="COUNTRY_REGION",
        title="Cases per Million vs Deaths per Million",
        log_x=True, log_y=True,
        color_continuous_scale='Viridis',
        size_max=40,
        template=FIG_TEMPLATE
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig

def build_cfr_bar_figure(filtered_df: pd.DataFrame) -> go.Figure:
    if filtered_df.empty:
        return go.Figure(layout=dict(template=FIG_TEMPLATE, title="Top 20 Countries by Case Fatality Rate (%)"))
    top = filtered_df.sort_values('CFR_%', ascending=False).head(20)
    fig = px.bar(
        top,
        x='COUNTRY_REGION', y='CFR_%',
        title="Top 20 Countries by Case Fatality Rate (%)",
        labels={"CFR_%": "Case Fatality Rate (%)", "COUNTRY_REGION": "Country"},
        text='CFR_%',
        template=FIG_TEMPLATE
    )
    ymax = (filtered_df['CFR_%'].max() * 1.2) if not filtered_df['CFR_%'].isna().all() else 1
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(
        yaxis_range=[0, ymax],
        xaxis_tickangle=-45,
        margin=dict(l=10, r=10, t=60, b=10)
    )
    return fig

def build_population_density_figure(filtered_df: pd.DataFrame) -> go.Figure:
    fd = fit_population_density_model(filtered_df)
    fig = px.scatter(
        fd,
        x='Population_Density_Calc', y='Cases_per_Million',
        hover_name='COUNTRY_REGION',
        title="Population Density vs Cases per Million",
        log_x=True, log_y=True,
        labels={
            'Population_Density_Calc': 'Population Density (people/km², log)',
            'Cases_per_Million': 'Cases per Million (log)'
        },
        template=FIG_TEMPLATE
    )
    fd_sorted = fd[['Population_Density_Calc', 'fitted']].dropna().sort_values('Population_Density_Calc')
    if not fd_sorted.empty:
        fig.add_traces(px.line(fd_sorted, x='Population_Density_Calc', y='fitted', template=FIG_TEMPLATE).data)
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig

def build_forecast_figure(selected_country: str, horizon_days: int) -> go.Figure:
    if not selected_country:
        return go.Figure(layout=dict(template=FIG_TEMPLATE, title="No country selected"))
    df_hist = fetch_historical_data(selected_country)
    forecast = make_forecast(df_hist, horizon_days=horizon_days)
    if df_hist.empty or forecast.empty:
        title = "Forecast unavailable (missing data or Prophet not installed)"
        return go.Figure(layout=dict(template=FIG_TEMPLATE, title=title))
    fig = px.line(
        forecast, x='ds', y='yhat',
        title=f"{horizon_days}-day Forecast of COVID-19 Cases — {selected_country}",
        labels={'ds': 'Date', 'yhat': 'Predicted Cases'},
        template=FIG_TEMPLATE
    )
    # Add uncertainty intervals
    fig.add_traces([
        go.Scatter(
            x=forecast['ds'], y=forecast['yhat_upper'],
            mode='lines', name='Upper', opacity=0.3, showlegend=False
        ),
        go.Scatter(
            x=forecast['ds'], y=forecast['yhat_lower'],
            mode='lines', name='Lower', opacity=0.3, fill='tonexty', showlegend=False
        )
    ])
    # Actuals
    fig.add_scatter(
        x=df_hist['ds'], y=df_hist['y'],
        mode='markers', name='Actual Cases', opacity=0.5
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig

def build_clusters_figure(k: int) -> go.Figure:
    cluster_df = fetch_clusters(k=k)
    if cluster_df.empty:
        return go.Figure(layout=dict(template=FIG_TEMPLATE, title="No clustering data available"))

    merged_df = pd.merge(df, cluster_df, how='left', on='COUNTRY_REGION')
    merged_df = merged_df.dropna(subset=['cluster'])
    fig = px.scatter(
        merged_df,
        x='Cases_per_Million',
        y='Deaths_per_Million',
        color=merged_df['cluster'].astype(str),
        size='Population_2020',
        hover_name='COUNTRY_REGION',
        title=f"COVID-19 Country Clusters (k={k})",
        log_x=True, log_y=True,
        size_max=40,
        labels={'color': 'Cluster'},
        template=FIG_TEMPLATE
    )
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), legend_title_text="Cluster")
    return fig

# ==================================
# Dash App Initialization
# ==================================
external_stylesheets = [dbc.themes.CYBORG if FIG_TEMPLATE == "plotly_dark" else dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "COVID-19 Interactive Dashboard"
server = app.server  # for deployment platforms (e.g., gunicorn)

# ==================================
# Layout
# ==================================
def card(title, body):
    return dbc.Card(
        [
            dbc.CardHeader(html.H5(title, className="mb-0")),
            dbc.CardBody(body)
        ],
        className="mb-3 shadow-sm",
        style={"borderRadius": "1rem"}
    )

app.layout = dbc.Container(fluid=True, children=[
    # Header
    dbc.Row([
        dbc.Col(html.H2("COVID-19 Interactive Dashboard", className="my-3"), width=12)
    ]),

    # Controls + Comments
    dbc.Row([
        # Controls Panel
        dbc.Col(
            card("Controls", [
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Countries", className="fw-bold"),
                        dcc.Dropdown(
                            id='country-dropdown',
                            options=[{'label': c, 'value': c} for c in countries],
                            multi=True,
                            placeholder="Start typing to search countries...",
                            style={"marginBottom": "1rem"}
                        )
                    ], width=12)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Forecast Horizon (days)", className="fw-bold"),
                        dcc.Slider(
                            id='horizon-slider',
                            min=7, max=60, step=1, value=30,
                            marks={7: "7", 14: "14", 30: "30", 45: "45", 60: "60"}
                        )
                    ], width=12, className="mb-3")
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Number of Clusters (k)", className="fw-bold"),
                        dcc.Slider(
                            id='k-slider',
                            min=2, max=8, step=1, value=3,
                            marks={i: str(i) for i in range(2, 9)}
                        )
                    ], width=12)
                ])
            ]),
            width=12, lg=4
        ),

        # Comments Panel
        dbc.Col(
            card("Annotations & Discussion", [
                html.Div([
                    dcc.Textarea(
                        id='comment-input',
                        placeholder='Share an insight or hypothesis about the selected countries...',
                        style={'width': '100%', 'height': 80}
                    ),
                    dbc.Button("Submit Comment", id='submit-comment', color="primary", className="mt-2"),
                    html.Div(id='comment-status', className="mt-2", style={"fontSize": "0.9rem", "opacity": 0.85})
                ], className="mb-3"),
                html.H6("Recent Comments", className="fw-bold"),
                html.Div(
                    id='comments-container',
                    style={"maxHeight": "320px", "overflowY": "auto", "paddingRight": "0.5rem"}
                )
            ]),
            width=12, lg=8
        ),
    ], className="g-3"),

    # Tabs for Analysis
    dbc.Row([
        dbc.Col(
            card("Analysis", [
                dcc.Tabs(
                    id="tabs", value="tab-overview",
                    children=[
                        dcc.Tab(label="Overview", value="tab-overview"),
                        dcc.Tab(label="Population Analysis", value="tab-population"),
                        dcc.Tab(label="Forecast", value="tab-forecast"),
                        dcc.Tab(label="Clusters", value="tab-clusters"),
                    ]
                ),
                html.Div(id="tabs-content", style={"paddingTop": "1rem"})
            ]),
            width=12
        )
    ]),

    html.Div(id="invisible-store", style={"display": "none"})  # placeholder
], className="pb-4")

# ==================================
# Compute Figures
# ==================================
@app.callback(
    Output('invisible-store', 'children'),
    Output('comment-status', 'children'),
    Output('tabs-content', 'children'),
    Input('country-dropdown', 'value'),
    Input('horizon-slider', 'value'),
    Input('k-slider', 'value'),
    Input('tabs', 'value'),
    Input('submit-comment', 'n_clicks'),
    State('comment-input', 'value')
)
def update_dashboard(selected_countries, horizon_days, k, tab_value, n_clicks, comment_text):
    """
    Single callback to:
    - Build figures based on controls
    - Handle comment submission
    - Render tab content with loading placeholders
    """
    # Normalize selection
    filtered_df = df if not selected_countries else df[df['COUNTRY_REGION'].isin(selected_countries)].copy()

    # --- Build Figures ---
    fig_cases_vs_deaths = build_cases_vs_deaths_figure(filtered_df)
    fig_cfr = build_cfr_bar_figure(filtered_df)
    fig_population = build_population_density_figure(filtered_df)

    first_country = selected_countries[0] if selected_countries else None
    fig_forecast = build_forecast_figure(first_country, horizon_days=horizon_days)
    fig_clusters = build_clusters_figure(k=int(k))

    # --- Comments: POST if needed ---
    status_msg = ""
    ctx = callback_context
    if ctx.triggered:
        triggered = ctx.triggered[0]['prop_id']
        if triggered == 'submit-comment.n_clicks' and n_clicks and n_clicks > 0:
            if comment_text and selected_countries:
                # Post one comment per selected country
                successes, failures = 0, 0
                for country in selected_countries:
                    payload = {
                        "country_key": country.lower(),
                        "comment_text": comment_text,
                        "user": "dash_user"
                    }
                    try:
                        resp = requests.post(f"{API_BASE}/comments/", json=payload, timeout=6)
                        resp.raise_for_status()
                        successes += 1
                    except Exception as e:
                        failures += 1
                        print(f"Error posting comment for {country}: {e}")
                status_msg = f"Posted {successes} comment(s)."
                if failures:
                    status_msg += f" {failures} failed."
            else:
                status_msg = "Please select at least one country and enter a comment."

    # --- Comments: GET and render ---
    comments_output = []
    def comment_bubble(user, text):
        return dbc.Alert(
            [html.Strong(f"{user}: "), html.Span(text)],
            color="secondary", className="py-2 px-3 my-1", style={"opacity": 0.9}
        )

    if selected_countries:
        for country in selected_countries:
            try:
                resp = requests.get(f"{API_BASE}/comments/", params={"country_key": country.lower()}, timeout=6)
                resp.raise_for_status()
                comments_data = resp.json().get('comments', [])
                if comments_data:
                    comments_output.append(html.H6(f"{country}", className="mt-3 mb-2"))
                    for c in comments_data:
                        comments_output.append(comment_bubble(c.get('user', 'user'), c.get('comment_text', '')))
            except Exception as e:
                print(f"Error fetching comments for {country}: {e}")
    else:
        comments_output.append(html.Div("Select one or more countries to see comments.", style={"opacity": 0.8}))

    # --- Tabs Content ---
    def loading_graph(graph_id, fig):
        return dcc.Loading(
            type="dot",
            children=dcc.Graph(id=graph_id, figure=fig, config={"displayModeBar": True}),
            color=None
        )

    if tab_value == "tab-overview":
        tabs_children = dbc.Row([
            dbc.Col(loading_graph('fig-cases-vs-deaths', fig_cases_vs_deaths), width=12, lg=6),
            dbc.Col(loading_graph('fig-cfr', fig_cfr), width=12, lg=6),
        ], className="g-3")
    elif tab_value == "tab-population":
        tabs_children = dbc.Row([
            dbc.Col(loading_graph('fig-population-density', fig_population), width=12)
        ], className="g-3")
    elif tab_value == "tab-forecast":
        tabs_children = dbc.Row([
            dbc.Col(loading_graph('fig-forecast', fig_forecast), width=12)
        ], className="g-3")
    elif tab_value == "tab-clusters":
        tabs_children = dbc.Row([
            dbc.Col(loading_graph('fig-clusters', fig_clusters), width=12)
        ], className="g-3")
    else:
        tabs_children = html.Div("Select a tab to view content.")

    # Put comments into the container
    app._cached_comments = comments_output  # cache for debugging if needed
    comments_container_children = comments_output

    # Return invisible_store (no-op), status message, and the tab page
    return "", status_msg, tabs_children

# Keep comment list updated when user changes selections, horizon, or k
@app.callback(
    Output('comments-container', 'children'),
    Input('country-dropdown', 'value'),
    Input('submit-comment', 'n_clicks'),
    State('comment-input', 'value')
)
def refresh_comments(selected_countries, n_clicks, comment_text):
    # This callback just mirrors the comments built in the main callback
    # to ensure the panel updates, even if tabs don't change
    try:
        return getattr(app, "_cached_comments")
    except Exception:
        return [html.Div("Select one or more countries to see comments.", style={"opacity": 0.8})]

# ==================================
# Entrypoint
# ==================================
if __name__ == '__main__':
    # Host/port can be overridden with env vars if needed
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", "8050")))
