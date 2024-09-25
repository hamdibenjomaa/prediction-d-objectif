import pandas as pd
import numpy as np
from prophet import Prophet
import pickle
import re

# Load and preprocess data
df_obj = pd.read_csv('Objrep.csv', delimiter=';', low_memory=False)

def preprocess_data(df):
    df['Objrep'] = df['Objrep'].str.replace(',', '.')
    df['Objrep'] = pd.to_numeric(df['Objrep'])
    return df

df_obj = preprocess_data(df_obj)
df_obj['date'] = pd.to_datetime(df_obj['Année'].astype(str) + '-' + df_obj['Mois'].astype(str) + '-' + df_obj['Jour'].astype(str))
df_obj.drop(['Activité', 'Jour', 'Mois', 'Année'], axis=1, inplace=True)

def sanitize_filename(filename):
    return re.sub(r'[^A-Za-z0-9_\-]', '_', filename)

def train_and_save_model(df_obj, Nom_Rep, Nom_Article, periods=12, min_data_points=1):
    df_filtered = df_obj[(df_obj['Nom_Rep'] == Nom_Rep) & (df_obj['Nom_Article'] == Nom_Article)]

    if len(df_filtered) < min_data_points or df_filtered['Objrep'].notnull().sum() < 2:
        print(f"Not enough data for {Nom_Rep} - {Nom_Article}. Skipping prediction.")
        return None

    df_prophet = df_filtered.rename(columns={'date': 'ds', 'Objrep': 'y'})

    model = Prophet(interval_width=0.95, daily_seasonality=True)
    model.fit(df_prophet)

    sanitized_rep = sanitize_filename(Nom_Rep)
    sanitized_art = sanitize_filename(Nom_Article)

    model_filename = f'model_{sanitized_rep}_{sanitized_art}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

# Train and save models for different representatives and articles
representatives = df_obj['Nom_Rep'].unique()
articles = df_obj['Nom_Article'].unique()

for rep in representatives:
    for art in articles:
        train_and_save_model(df_obj, rep, art)
