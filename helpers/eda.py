import pandas as pd
import plotly.express as px

# Dynamic date column detection
def detect_date_column(df):
    for col in df.columns:
        if "date" in col.lower():
            return col
    return None

# General column detection for keywords
def detect_column(df, possible_names):
    for name in possible_names:
        for col in df.columns:
            if name.lower() in col.lower():
                return col
    return None

# Clean and enrich data dynamically
def clean_data(df):
    df = df.copy()

    # Detect and convert date column
    date_col = detect_date_column(df)
    if not date_col:
        raise ValueError("No date-like column found in dataset.")
    
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    df = df.rename(columns={date_col: "InvoiceDate"})

    # Remove cancelled transactions if InvoiceNo exists
    if 'InvoiceNo' in df.columns:
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

    # Detect quantity and price columns
    quantity_col = detect_column(df, ['Quantity', 'qty'])
    price_col = detect_column(df, ['UnitPrice', 'Price', 'SellingPrice', 'actprice1'])

    if quantity_col and price_col:
        df['Sales'] = df[quantity_col] * df[price_col]
        df = df[df[quantity_col] > 0]
    else:
        df['Sales'] = 1  # fallback

    df = df.drop_duplicates()
    return df

# Sales over time (grouped by day)
def get_sales_over_time(df):
    if 'InvoiceDate' not in df.columns or 'Sales' not in df.columns:
        return pd.DataFrame()
    
    df['Date'] = df['InvoiceDate'].dt.date
    daily_sales = df.groupby('Date')['Sales'].sum().reset_index()
    daily_sales['Date'] = pd.to_datetime(daily_sales['Date'])
    return daily_sales

def plot_sales_over_time(df):
    if df.empty:
        return px.line(title="No Sales Data Found")
    return px.line(df, x='Date', y='Sales', title='Sales Over Time')

# Top-selling products
def top_products(df, top_n=10):
    desc_col = detect_column(df, ['Description', 'title', 'Product', 'Item'])
    if not desc_col or 'Sales' not in df.columns:
        return pd.DataFrame()
    
    top = df.groupby(desc_col)['Sales'].sum().reset_index()
    top = top.sort_values(by='Sales', ascending=False).head(top_n)
    top = top.rename(columns={desc_col: 'Product'})
    return top

def plot_top_products(df):
    if df.empty:
        return px.bar(title="No Product Data Found")
    return px.bar(df, x='Product', y='Sales', title='Top Products', text='Sales')

# Sales by country
def sales_by_country(df):
    country_col = detect_column(df, ['Country', 'Location', 'Region'])
    if not country_col or 'Sales' not in df.columns:
        return pd.DataFrame()
    
    country_sales = df.groupby(country_col)['Sales'].sum().reset_index()
    country_sales = country_sales.rename(columns={country_col: 'Country'})
    country_sales = country_sales.sort_values(by='Sales', ascending=False)
    return country_sales

def plot_sales_by_country(df):
    if df.empty:
        return px.bar(title="No Country Data Found")
    return px.bar(df, x='Country', y='Sales', title='Sales by Country', text='Sales')
