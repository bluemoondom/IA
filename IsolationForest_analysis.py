# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 13:06:02 2025

@author: dominika
"""

import pymssql
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

DB_CONFIG = {
        'server': 'server',
        'database': 'database',
        'username': 'username', 
        'password': 'password'
    }
SQL_QUERY = """
    SELECT  AS DatumPripad,  AS Castka
    FROM 
    WHERE 
    ORDER BY  DESC
    """

class mSQL:
    def __init__(self, server, database, username, password):
        self.server = server
        self.database = database
        self.username = username
        self.password = password
        self.connection = None
    
    def connect(self):
        try:
            self.connection = pymssql.connect(
                server=self.server,
                user=self.username,
                password=self.password,
                database=self.database
            )
            return True
        except Exception as e:
            print(f"Chyba připojení: {e}")
            return False
    
    def load_data(self, sql_query):
        if not self.connection:
            if not self.connect():
                return None
        try:
            df = pd.read_sql(sql_query, self.connection)
            print(f"Načteno {len(df)} řádků")
            return df
        except Exception as e:
            print(f"Chyba při načítání dat: {e}")
            return None
    
    def close(self):
        if self.connection:
            self.connection.close()

def prepare_sql_data(df):
    data = df.copy()
    if 'DatumPripad' in data.columns:
        data['DatumPripad'] = pd.to_datetime(data['DatumPripad'])
    data = data.sort_values('DatumPripad').reset_index(drop=True)
    data = data.dropna(subset=['DatumPripad', 'Castka'])
    print(f"Od: {data['DatumPripad'].min()} Do {data['DatumPripad'].max()}")
    print(f"Od: {data['Castka'].min():.2f} Do {data['Castka'].max():.2f}")
    return data

def detect_ml_anomalies(data, contamination=0.01):
    data['day_of_week'] = data['DatumPripad'].dt.dayofweek
    data['day_of_month'] = data['DatumPripad'].dt.day
    data['month'] = data['DatumPripad'].dt.month
    data['rolling_mean_7d'] = data['Castka'].rolling(window=7, center=True).mean()
    data['rolling_std_7d'] = data['Castka'].rolling(window=7, center=True).std()
    feature_columns = ['Castka', 'day_of_week', 'day_of_month', 'month', 'rolling_mean_7d', 'rolling_std_7d']
    X = data[feature_columns].fillna(data[feature_columns].mean())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomalies = iso_forest.fit_predict(X_scaled)
    data['anomaly_ml'] = (anomalies == -1).astype(int)
    print(f"ML: nalezeno {(anomalies == -1).sum()} anomálií")
    return data, iso_forest

def plot_anomalies(data):
    fig, axes = plt.subplots(1, 1, figsize=(15, 12))
    axes.scatter(data['DatumPripad'], data['Castka'], alpha=0.6, s=20, label='Normální případy')
    
    ml_anomalies = data[data['anomaly_ml'] == 1]
    if len(ml_anomalies) > 0:
        axes.scatter(ml_anomalies['DatumPripad'], ml_anomalies['Castka'],
                       color='purple', s=50, label=f'ML anomálie ({len(ml_anomalies)})', zorder=5)
    
    axes.set_title('ML Anomalies Detection (Isolation Forest)')
    axes.set_ylabel('Částka')
    axes.legend()
    axes.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


detector = mSQL(**DB_CONFIG)
detector.connect()

data = detector.load_data(SQL_QUERY)
data = prepare_sql_data(data)
data, ml_model = detect_ml_anomalies(data)
plot_anomalies(data)

anomalies_summary = data[
    (data['anomaly_ml'] == 1)
][['DatumPripad', 'Castka', 'anomaly_ml']]
    
print(f"Nalezeno celkem {len(anomalies_summary)} anomálních záznamů:")
print(anomalies_summary.head(10).to_string(index=False))

anomalies_summary.to_csv('anomalie_sql.csv', index=False)
detector.close()


