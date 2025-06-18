import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
import io

def prepare_forecast_data(df):
    daily_sales = df.groupby(df['InvoiceDate'].dt.date).agg({'Sales':'sum'}).reset_index()
    daily_sales.columns = ['ds', 'y']
    return daily_sales

def train_forecast_model(daily_sales):
    model = Prophet(daily_seasonality=True)
    model.fit(daily_sales)

    future = model.make_future_dataframe(periods=30)  # Forecast 30 days
    forecast = model.predict(future)
    return model, forecast

def plot_forecast(model, forecast):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=model.history['ds'], y=model.history['y'], mode='markers', name='Historical'))
    fig.update_layout(title='Sales Forecast', xaxis_title='Date', yaxis_title='Sales')
    return fig

def export_forecast_to_excel(forecast):
    output = io.BytesIO()
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_excel(output, index=False)
    processed_data = output.getvalue()
    return processed_data
