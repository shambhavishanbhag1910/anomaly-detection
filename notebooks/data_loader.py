import os
import pandas as pd

def load_clean_data(file_name="D:/Deep Learning/Anomaly Detection/data/predictive_maintenance.csv"):
    """
    Finds the data directory, loads the CSV, and returns a pandas DataFrame.
    """
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", file_name)
    
    print(f"Loading data from: {data_path}")
    
    df = pd.read_csv(data_path)
    
    return df