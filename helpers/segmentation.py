import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px

def calculate_rfm(df, current_date):
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (current_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'Sales': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    return rfm

def rfm_segmentation(rfm):
    # Log transform Monetary
    rfm['Monetary_log'] = np.log1p(rfm['Monetary'])
    # Normalize features
    rfm_normalized = rfm[['Recency', 'Frequency', 'Monetary_log']]
    rfm_normalized = (rfm_normalized - rfm_normalized.mean()) / rfm_normalized.std()

    # KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm['Segment'] = kmeans.fit_predict(rfm_normalized)
    return rfm

def plot_rfm_segments(rfm):
    fig = px.scatter_3d(rfm, x='Recency', y='Frequency', z='Monetary', color='Segment',
                        title='Customer Segments (RFM + KMeans)')
    return fig
