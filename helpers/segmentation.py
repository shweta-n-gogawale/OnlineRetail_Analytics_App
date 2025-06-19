import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px

# Auto-detect relevant columns
def detect_customer_column(df):
    for col in df.columns:
        if "customer" in col.lower():
            return col
    return None

def detect_date_column(df):
    for col in df.columns:
        if "date" in col.lower():
            return col
    return None

def detect_invoice_column(df):
    for col in df.columns:
        if "invoice" in col.lower():
            return col
    return None

# Calculate RFM metrics
def calculate_rfm(df, current_date):
    df = df.copy()

    cust_col = detect_customer_column(df)
    date_col = detect_date_column(df)
    invoice_col = detect_invoice_column(df)

    if not all([cust_col, date_col, invoice_col]) or 'Sales' not in df.columns:
        return pd.DataFrame()

    # Parse date and drop invalid rows
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[cust_col, date_col])

    # Group and calculate RFM
    rfm = df.groupby(cust_col).agg({
        date_col: lambda x: (current_date - x.max()).days,
        invoice_col: 'nunique',
        'Sales': 'sum'
    }).reset_index()

    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm

# Perform clustering using KMeans
def rfm_segmentation(rfm):
    if rfm.empty:
        return pd.DataFrame()

    # Transform for skewed Monetary values
    rfm['Monetary_log'] = np.log1p(rfm['Monetary'])

    # Normalize
    rfm_norm = rfm[['Recency', 'Frequency', 'Monetary_log']]
    rfm_norm = (rfm_norm - rfm_norm.mean()) / rfm_norm.std()

    # KMeans Clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Segment'] = kmeans.fit_predict(rfm_norm)
    return rfm

# Plot the RFM 3D segments
def plot_rfm_segments(rfm):
    if rfm.empty or not all(col in rfm.columns for col in ['Recency', 'Frequency', 'Monetary', 'Segment']):
        return px.scatter_3d(title="Insufficient RFM data for segmentation")
    
    fig = px.scatter_3d(
        rfm,
        x='Recency', y='Frequency', z='Monetary',
        color='Segment',
        title='Customer Segments (RFM + KMeans)',
        opacity=0.8
    )
    return fig
