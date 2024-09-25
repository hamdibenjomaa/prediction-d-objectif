from flask import Flask, request, render_template
import pickle
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import io
import base64
import re

app = Flask(__name__)

# Load and preprocess data
df_obj = pd.read_csv('Objrep.csv', delimiter=';', low_memory=False)

def preprocess_data(df):
    df['Objrep'] = df['Objrep'].str.replace(',', '.')
    df['Objrep'] = pd.to_numeric(df['Objrep'])
    df['date'] = pd.to_datetime(df['Année'].astype(str) + '-' + df['Mois'].astype(str) + '-' + df['Jour'].astype(str))
    df.drop(['Activité', 'Jour', 'Mois', 'Année'], axis=1, inplace=True)
    return df

df_obj = preprocess_data(df_obj)

# Get unique representatives and articles
representatives = df_obj['Nom_Rep'].unique()
articles = df_obj['Nom_Article'].unique()

def sanitize_filename(filename):
    return re.sub(r'[^A-Za-z0-9_\-]', '_', filename)

# Load the model based on representative and article
def load_model(Nom_Rep, Nom_Article):
    Nom_Rep = sanitize_filename(Nom_Rep)
    Nom_Article = sanitize_filename(Nom_Article)
    model_filename = f'model_{Nom_Rep}_{Nom_Article}.pkl'
    print(f"Attempting to load model from: {model_filename}")
    try:
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print(f"Model file {model_filename} not found.")
        return None

@app.route('/')
def home():
    return render_template("forest_fire.html", representatives=representatives, articles=articles)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        Nom_Rep = request.form.get('Representant')
        Nom_Article = request.form.get('Article')

        print(f"Received values - Representative: {Nom_Rep}, Article: {Nom_Article}")

        model = load_model(Nom_Rep, Nom_Article)
        if model is None:
            return render_template('forest_fire.html', pred=f'No model found for Representative: {Nom_Rep} and Article: {Nom_Article}', representatives=representatives, articles=articles)

        # Create future dates for prediction
        future = model.make_future_dataframe(periods=12, freq='M')

        # Forecast
        forecast = model.predict(future)

        # Ensure yhat is non-negative
        forecast['yhat'] = forecast['yhat'].clip(lower=0)
        forecast['yhat_upper'] = forecast['yhat_upper'].clip(lower=0)
        forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)

        last_prediction = forecast[['ds', 'yhat']].tail(1)
        prediction = last_prediction['yhat'].values[0]

        # Prepare table data
        forecast_table = forecast[['ds', 'yhat', 'yhat_upper', 'yhat_lower']].tail(12)
        forecast_table['ds'] = forecast_table['ds'].dt.strftime('01-%m-%Y')  # Format date

        # Plot the forecast
        img = io.BytesIO()
        fig, ax = plt.subplots(figsize=(10, 6))
        model.plot(forecast, ax=ax)
        ax.scatter(model.history['ds'], model.history['y'], color='red', label='Actual Sales')
        ax.set_title(f'Forecast for {Nom_Rep} - {Nom_Article}', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Sales Objective', fontsize=14)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

        return render_template('forest_fire.html', pred=f'The predicted sales objective is {prediction:.2f}', plot_url=plot_url, table=forecast_table.to_html(classes='table table-striped', index=False), representatives=representatives, articles=articles)

    except Exception as e:
        print("Error:", e)
        return render_template('forest_fire.html', pred='Error in prediction. Please check the inputs.', representatives=representatives, articles=articles)

if __name__ == '__main__':
    app.run(debug=True)
