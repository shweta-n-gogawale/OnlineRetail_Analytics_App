import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
import io
from fpdf import FPDF
import tempfile
import os

# Auto-detect date column
def detect_date_column(df):
    for col in df.columns:
        if "date" in col.lower():
            return col
    return None

# Prepare time series data for Prophet
def prepare_forecast_data(df):
    df = df.copy()
    date_col = detect_date_column(df)

    if not date_col or 'Sales' not in df.columns:
        return pd.DataFrame()

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.rename(columns={date_col: 'InvoiceDate'})

    daily_sales = df.groupby(df['InvoiceDate'].dt.date)['Sales'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    return daily_sales

# Train Prophet model
def train_forecast_model(daily_sales):
    if daily_sales.empty:
        return None, pd.DataFrame()

    model = Prophet(daily_seasonality=True)
    model.fit(daily_sales)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return model, forecast

# Forecast plot
def plot_forecast(model, forecast):
    if forecast.empty or model is None:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=model.history['ds'], y=model.history['y'], mode='markers', name='Historical'))

    fig.update_layout(title='📈 30-Day Sales Forecast',
                      xaxis_title='Date', yaxis_title='Sales',
                      template='plotly_white')
    return fig

# Excel export
def export_forecast_to_excel(forecast):
    if forecast.empty:
        return None

    output = io.BytesIO()
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_excel(output, index=False)
    return output.getvalue()

# NEW: Export forecast to PDF
def export_forecast_to_pdf(forecast):
    if forecast.empty:
        return None

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="30-Day Sales Forecast Report", ln=True, align='C')
    pdf.ln(10)

    # Limit to first 20 rows for PDF readability
    limited = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(20).round(2)

    for index, row in limited.iterrows():
        line = f"{row['ds'].date()} | Prediction: {row['yhat']} | Range: ({row['yhat_lower']}, {row['yhat_upper']})"
        pdf.cell(200, 10, txt=line, ln=True)

    # Save to temp file
    tmp_path = tempfile.mktemp(suffix=".pdf")
    pdf.output(tmp_path)
    return tmp_path
