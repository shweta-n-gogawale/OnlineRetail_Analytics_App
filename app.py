import streamlit as st
from utils import load_data
from helpers import eda, forecasting, segmentation
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="📊 Online Retail Sales Analytics", layout="wide")

st.title("📊 Online Retail Sales Analytics & Forecasting")

menu = ["Home", "Load Data", "EDA", "Forecast", "Segmentation", "About"]
choice = st.sidebar.radio("Navigate", menu)

# Global df container
if 'df' not in st.session_state:
    st.session_state.df = None

if choice == "Home":
    st.markdown("""
    ### Welcome to the Online Retail Sales Analytics Dashboard
    - Load dataset from Excel or CSV file
    - Perform exploratory data analysis (EDA)
    - Forecast sales using Prophet
    - Segment customers using RFM + KMeans
    """)

elif choice == "Load Data":
    st.subheader("Load Dataset")
    uploaded_file = st.file_uploader("Upload Online Retail file (.xlsx or .csv)", type=['xlsx', 'csv'])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.session_state.df = df
        st.success("Dataset loaded successfully!")
        st.write(df.head())

elif choice == "EDA":
    st.subheader("Exploratory Data Analysis")
    if st.session_state.df is None:
        st.warning("Please load data first on 'Load Data' tab.")
    else:
        df_clean = eda.clean_data(st.session_state.df)

        st.markdown("### Sales Over Time")
        sales_time = eda.get_sales_over_time(df_clean)
        st.plotly_chart(eda.plot_sales_over_time(sales_time), use_container_width=True)

        st.markdown("### Top Products")
        top_prod = eda.top_products(df_clean)
        st.plotly_chart(eda.plot_top_products(top_prod), use_container_width=True)

        st.markdown("### Sales by Country")
        sales_country = eda.sales_by_country(df_clean)
        st.plotly_chart(eda.plot_sales_by_country(sales_country), use_container_width=True)

elif choice == "Forecast":
    st.subheader("Sales Forecasting")
    if st.session_state.df is None:
        st.warning("Please load data first on 'Load Data' tab.")
    else:
        df_clean = eda.clean_data(st.session_state.df)
        daily_sales = forecasting.prepare_forecast_data(df_clean)

        with st.spinner("Training forecasting model..."):
            model, forecast = forecasting.train_forecast_model(daily_sales)

        st.success("Forecast generated!")
        st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        st.plotly_chart(forecasting.plot_forecast(model, forecast), use_container_width=True)

        st.download_button("Download Forecast Excel",
                           data=forecasting.export_forecast_to_excel(forecast),
                           file_name="sales_forecast.xlsx")

elif choice == "Segmentation":
    st.subheader("Customer Segmentation (RFM + Clustering)")
    if st.session_state.df is None:
        st.warning("Please load data first on 'Load Data' tab.")
    else:
        df_clean = eda.clean_data(st.session_state.df)
        current_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = segmentation.calculate_rfm(df_clean, current_date)
        segmented = segmentation.rfm_segmentation(rfm)

        st.dataframe(segmented.head())

        fig = segmentation.plot_rfm_segments(segmented)
        st.plotly_chart(fig, use_container_width=True)

elif choice == "About":
    st.markdown("""
    ### About
    Developed by Shweta Gogawale  
    This dashboard uses the UCI Online Retail dataset for high-level sales analytics.
    """)