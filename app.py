import streamlit as st
from utils import load_data, detect_date_column, detect_sales_column
from helpers import eda, forecasting, segmentation
import pandas as pd
from datetime import datetime
from io import BytesIO
from fpdf import FPDF

# --- Streamlit App Config ---
st.set_page_config(page_title="📊 Retail Analytics Dashboard", layout="wide", page_icon="🛍️")

# --- Custom CSS for UI + Spinner Animation ---
st.markdown("""
    <style>
        /* Global fonts and layout */
        .main-heading {
            font-size: 2.6rem;
            font-weight: 800;
            color: #1f2937;
            text-align: center;
            margin-bottom: 1rem;
        }
        .card {
            background-color: #ffffff;
            padding: 1.6rem;
            margin: 1.2rem 0;
            border-radius: 16px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            transition: all 0.3s ease-in-out;
        }
        .card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.12);
        }
        ul {
            padding-left: 1.2rem;
        }
        ul li {
            margin-bottom: 0.4rem;
        }
        .stButton>button {
            border-radius: 10px;
            padding: 0.5rem 1.5rem;
            font-weight: 600;
            background-color: #4F46E5;
            color: white;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #4338CA;
            transform: scale(1.05);
        }
        .block-container {
            padding-top: 2rem;
            max-width: 1200px;
        }
        .sidebar .sidebar-content {
            background-color: #f3f4f6;
        }

        /* Animated spinner */
        @keyframes spin {
            to {transform: rotate(360deg);}
        }
        .loading-spinner {
            border: 5px solid #f3f3f3;
            border-top: 5px solid #4F46E5;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: auto;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<div class='main-heading'>📊 Retail Analytics & Forecasting Dashboard</div>", unsafe_allow_html=True)

# --- Sidebar Navigation ---
menu = ["🏠 Home", "📥 Load Data", "📊 EDA", "📈 Forecast", "👥 Segmentation", "ℹ️ About"]
choice = st.sidebar.radio("Navigate", menu)

# --- Global Data ---
if 'df' not in st.session_state:
    st.session_state.df = None

# --- Home Page ---
if choice == "🏠 Home":
    st.markdown("""
    <div class='card'>
        <h2>👋 Welcome!</h2>
        <p>This dashboard helps you explore and analyze your retail sales data with:</p>
        <ul>
            <li>📈 Visual EDA: trends, top products, sales by region</li>
            <li>🔮 Sales Forecasting using Facebook Prophet</li>
            <li>👥 Customer Segmentation (RFM + Clustering)</li>
            <li>📁 Upload Excel/CSV Files with auto-detection</li>
            <li>📤 Export outputs to Excel & PDF</li>
        </ul>
        <p>💡 Tip: Use sidebar to upload your data and navigate</p>
    </div>
    """, unsafe_allow_html=True)

# --- Load Data Page ---
elif choice == "📥 Load Data":
    st.markdown("""
    <div class='card'>
        <h3>📁 Upload Your Dataset</h3>
        <p>Supported formats: <code>.xlsx</code> or <code>.csv</code></p>
        <p>Ensure your file contains the following:</p>
        <ul>
            <li><strong>Date column</strong> (e.g., <code>InvoiceDate</code>)</li>
            <li><strong>Quantity</strong> & <strong>Price</strong> columns</li>
            <li><strong>CustomerID</strong> & <strong>InvoiceNo</strong> for segmentation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload Excel or CSV", type=["xlsx", "csv"])

    if uploaded_file:
        with st.spinner("🔄 Loading and processing your data..."):
            try:
                df = load_data(uploaded_file)
                st.session_state.df = df
                st.success("✅ Dataset loaded successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")

# --- EDA Page ---
elif choice == "📊 EDA":
    st.markdown("<div class='card'><h3>Exploratory Data Analysis</h3></div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first from the 'Load Data' tab.")
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

# --- Forecasting Page ---
elif choice == "📈 Forecast":
    st.markdown("<div class='card'><h3>Sales Forecasting</h3></div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first from the 'Load Data' tab.")
    else:
        df_clean = eda.clean_data(st.session_state.df)
        try:
            with st.spinner("📈 Training Prophet model..."):
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

# --- Segmentation Page ---
elif choice == "👥 Segmentation":
    st.markdown("<div class='card'><h3>Customer Segmentation</h3></div>", unsafe_allow_html=True)

    if st.session_state.df is None:
        st.warning("⚠️ Please load data first from the 'Load Data' tab.")
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

# --- About Page ---
elif choice == "ℹ️ About":
    st.markdown("""
    <div class='card'>
        <h3>👩‍💻 Developer: Shweta Gogawale</h3>
        <ul>
            <li>🎓 Computer Engineering Student, BVCOEW Pune</li>
            <li>💡 Passionate about Data Analytics & Dashboards</li>
            <li>📬 Email: gogawaleshweta12@gmail.com</li>
            <li>🔗 GitHub: <a href="https://github.com/shweta-n-gogawale" target="_blank">shweta-n-gogawale</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
