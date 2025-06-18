import pandas as pd

def load_data(file_path):
    # Read Excel because original dataset is .xlsx
    df = pd.read_excel(file_path, engine='openpyxl')
    return df
