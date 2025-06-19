import pandas as pd

# Load uploaded data (Excel or CSV)
def load_data(file):
    if file.name.endswith('.xlsx'):
        df = pd.read_excel(file, engine='openpyxl')
    elif file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        raise ValueError("Unsupported file format. Please upload a .csv or .xlsx file.")
    return df

# Auto-detect datetime column name
def detect_date_column(df):
    for col in df.columns:
        if "date" in col.lower():
            try:
                pd.to_datetime(df[col], errors='coerce')
                return col
            except:
                continue

    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().sum() > 0:
                return col
        except:
            continue
    return None

# Auto-detect sales-related column name
def detect_sales_column(df):
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['sales', 'amount', 'revenue', 'total', 'price']):
            if pd.api.types.is_numeric_dtype(df[col]):
                return col

    numeric_cols = df.select_dtypes(include='number').columns
    if len(numeric_cols) > 0:
        return numeric_cols[0]
    return None
