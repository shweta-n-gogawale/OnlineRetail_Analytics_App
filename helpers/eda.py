import pandas as pd
import plotly.express as px
import numpy as np

def clean_data(df):
    # Drop rows with missing CustomerID (needed for RFM)
    df = df.dropna(subset=['CustomerID'])

    # Remove canceled orders (InvoiceNo starting with 'C')
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

    # Convert InvoiceDate to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # Add a Sales column (Quantity * UnitPrice)
    df['Sales'] = df['Quantity'] * df['UnitPrice']

    # Remove negative or zero sales quantities
    df = df[df['Quantity'] > 0]

    return df

def get_sales_over_time(df):
    sales_time = df.groupby(df['InvoiceDate'].dt.to_period('D')).agg({'Sales': 'sum'}).reset_index()
    sales_time['InvoiceDate'] = sales_time['InvoiceDate'].dt.to_timestamp()
    return sales_time

def plot_sales_over_time(df):
    fig = px.line(df, x='InvoiceDate', y='Sales', title='Daily Sales Over Time')
    return fig

def top_products(df, top_n=10):
    prod_sales = df.groupby('Description').agg({'Sales':'sum', 'Quantity':'sum'}).reset_index()
    prod_sales = prod_sales.sort_values('Sales', ascending=False).head(top_n)
    return prod_sales

def plot_top_products(prod_sales):
    fig = px.bar(prod_sales, x='Description', y='Sales', title='Top Selling Products by Sales', text='Sales')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def sales_by_country(df):
    country_sales = df.groupby('Country').agg({'Sales':'sum'}).reset_index()
    country_sales = country_sales.sort_values('Sales', ascending=False)
    return country_sales

def plot_sales_by_country(country_sales):
    fig = px.bar(country_sales, x='Country', y='Sales', title='Sales by Country')
    return fig
