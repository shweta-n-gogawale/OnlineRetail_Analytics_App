# 🛍️ Online Retail Sales Analytics & Forecasting Dashboard

A professional, end-to-end **Streamlit dashboard** for analyzing and forecasting sales data from **any retail dataset** (CSV or Excel).  
Just upload your file — the app automatically cleans, analyzes, and visualizes your data in real time!

---

## ✨ Features

✅ Upload any `.xlsx` or `.csv` file — no hardcoded dataset  
✅ Auto-detection of **date**, **quantity**, **price**, **customer ID**, and other key columns  
✅ 🔍 EDA with interactive charts: sales trends, top products, and country-level insights  
✅ 🔮 Sales Forecasting using Facebook Prophet  
✅ 👥 Customer Segmentation with RFM + KMeans  
✅ 📤 Export forecast results to Excel and PDF  
✅ ⚡️ Modern UI with cards, hover animations, and dark/light mode  
✅ ☁️ Deployed live with **Streamlit Cloud**  

---

## 🛠️ Tech Stack

| Purpose           | Technology                     |
|------------------|---------------------------------|
| 🐍 Language       | Python                          |
| 🌐 Web Framework  | Streamlit                       |
| 📊 Analysis       | Pandas, NumPy                   |
| 📈 Visualization  | Plotly, Plotly Express          |
| 🔮 Forecasting    | Prophet (Facebook)              |
| 📌 Clustering     | Scikit-learn (KMeans)           |
| 📁 File Support   | openpyxl, csv                   |
| 📄 PDF Export     | FPDF, BytesIO                   |
| ☁️ Deployment     | Streamlit Cloud                 |

---
##📁 Supported Dataset Format
Upload any retail dataset in .csv or .xlsx format

The app dynamically detects required columns like:

Column Type	Example Names
📅 Date	InvoiceDate, OrderDate
📦 Quantity	Quantity, qty
💰 Price	UnitPrice, Price
🧾 Invoice	InvoiceNo
👤 Customer	CustomerID

No code edits needed — just upload your file in the sidebar and explore!

##🧠 App Modules
Section	Description
🏠 Home	Intro + upload instructions
📥 Load Data	Upload your Excel/CSV files dynamically
📊 EDA	Visualize sales trends, top items, and countries
📈 Forecast	30-day forecast using Prophet
👥 Segmentation	RFM Clustering with KMeans
ℹ️ About	Developer info and contact

##🔗 Live Demo
👉 Click to Try the Live App on Streamlit Cloud

##👩‍💻 Author
Shweta Gogawale
🎓 Computer Engineering Student — BVCOEW Pune
📧 gogawaleshweta12@gmail.com
🔗 GitHub: shweta-n-gogawale

##⚠️ Disclaimer
This is an academic and portfolio project only.

No external dataset is bundled or used by default.

Users are responsible for uploading their own retail data for analysis.

No personal data is stored or collected by this app.

## 🖥️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/shweta-n-gogawale/OnlineRetail_Analytics_App.git
cd OnlineRetail_Analytics_App

# 2. (Optional) Create virtual environment
python -m venv env
source env/bin/activate       # On Windows: env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
