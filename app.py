import streamlit as st
from utils import load_data, detect_date_column, detect_sales_column
from helpers import eda, forecasting, segmentation
import pandas as pd
from datetime import datetime
from io import BytesIO
from fpdf import FPDF

# Set Streamlit config
st.set_page_config(page_title="📊 Retail Analytics Dashboard", layout="wide", page_icon="🛍️")

# Custom CSS and animations for better UI
st.markdown("""
    <style>
        .main-heading {
            font-size: 2.5rem;
            font-weight: 700;
            color: #3A3A3A;
            text-align: center;
        }
        .card {
            background-color: #f9f9f9;
            padding: 1.5rem;
            margin: 1rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6;
        }
        .block-container {
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='main-heading'>📊 Retail Analytics & Forecasting Dashboard</div>", unsafe_allow_html=True)

# Navigation
menu = ["🏠 Home", "📥 Load Data", "📊 EDA", "📈 Forecast", "👥 Segmentation", "ℹ️ About"]
choice = st.sidebar.radio("Navigate", menu)

# Store global data
if 'df' not in st.session_state:
    st.session_state.df = None

# Home
if choice == "🏠 Home":
    st.markdown("""
        <div class='card'>
        <h3>Welcome!</h3>
        This dashboard helps you analyze retail sales data with features like:
        - 📊 Exploratory Data Analysis
        - 🔮 Forecasting using Prophet
        - 👥 Customer Segmentation
        - 📥 Excel & CSV file support
        - 📤 Export to Excel & PDF
        </div>
    """, unsafe_allow_html=True)

# Load Data
elif choice == "📥 Load Data":
    st.markdown("<div class='card'><h3>Upload Dataset</h3></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Retail Dataset (.xlsx or .csv)", type=["xlsx", "csv"])

    if uploaded_file:
        try:
            df = load_data(uploaded_file)
            st.session_state.df = df
            st.success("✅ Dataset loaded successfully!")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Error loading file: {e}")

# EDA
elif choice == "📊 EDA":
    st.markdown("<div class='card'><h3>Exploratory Data Analysis</h3></div>", unsafe_allow_html=True)
    if st.session_state.df is None:
        st.warning("⚠️ Please load data first on 'Load Data' tab.")
    else:
        df_clean = eda.clean_data(st.session_state.df)

        st.markdown("#### 📈 Sales Over Time")
        sales_time = eda.get_sales_over_time(df_clean)
        st.plotly_chart(eda.plot_sales_over_time(sales_time), use_container_width=True)

        st.markdown("#### 🏆 Top Products")
        top_prod = eda.top_products(df_clean)
        st.plotly_chart(eda.plot_top_products(top_prod), use_container_width=True)

        st.markdown("#### 🌍 Sales by Country")
        sales_country = eda.sales_by_country(df_clean)
        st.plotly_chart(eda.plot_sales_by_country(sales_country), use_container_width=True)

# Forecasting
elif choice == "📈 Forecast":
    st.markdown("<div class='card'><h3>Sales Forecasting</h3></div>", unsafe_allow_html=True)
    if st.session_state.df is None:
        st.warning("⚠️ Please load data first on 'Load Data' tab.")
    else:
        df_clean = eda.clean_data(st.session_state.df)
        try:
            daily_sales = forecasting.prepare_forecast_data(df_clean)
            model, forecast = forecasting.train_forecast_model(daily_sales)
            st.success("✅ Forecast generated!")
            st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
            st.plotly_chart(forecasting.plot_forecast(model, forecast), use_container_width=True)

            st.download_button("📥 Download Excel Forecast",
                               data=forecasting.export_forecast_to_excel(forecast),
                               file_name="forecast.xlsx")

            if hasattr(forecasting, "export_forecast_to_pdf"):
                pdf_path = forecasting.export_forecast_to_pdf(forecast)
                with open(pdf_path, "rb") as f:
                    st.download_button("📄 Download Forecast PDF", f, file_name="forecast.pdf")
        except Exception as e:
            st.error(f"❌ Forecasting failed: {e}")

# Segmentation
elif choice == "👥 Segmentation":
    st.markdown("<div class='card'><h3>Customer Segmentation</h3></div>", unsafe_allow_html=True)
    if st.session_state.df is None:
        st.warning("⚠️ Please load data first on 'Load Data' tab.")
    else:
        try:
            df_clean = eda.clean_data(st.session_state.df)
            current_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
            rfm = segmentation.calculate_rfm(df_clean, current_date)
            segmented = segmentation.rfm_segmentation(rfm)
            st.dataframe(segmented.head())

            fig = segmentation.plot_rfm_segments(segmented)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Segmentation failed: {e}")

# About
elif choice == "ℹ️ About":
    st.markdown("""
    ### 👩‍💻 Developer: Shweta Gogawale
    - 🎓 Computer Engineering Student, BVCOEW Pune
    - 💡 Passionate about data analytics and intelligent dashboards
    - 📬 Contact: gogawaleshweta12@gmail.com
    - 🔗 GitHub: [shweta-n-gogawale](https://github.com/shweta-n-gogawale)
    """)