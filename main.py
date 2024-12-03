import os
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib  # Importar joblib para cargar el scaler
from datetime import datetime, timedelta

# Inicializa la app Flask
app = Flask(__name__)

# Cargar el modelo preentrenado
model = load_model('./lstm-model/modelo_lstm.h5')

# Cargar el scaler guardado previamente (asegúrate de tener el archivo scaler.pkl)
scaler = joblib.load('scaler.pkl')  

# Cargar los datos históricos
historical_data = pd.read_csv('dfLimpio.csv')
historical_data['Date'] = pd.to_datetime(historical_data['Date'])
historical_data.set_index('Date', inplace=True)
historical_data = historical_data.sort_index(ascending=True)


scaled_data = scaler.transform(historical_data[['Adj Close']].values)

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos de la solicitud POST
        data = request.get_json()
        future_date_str = data.get('date')

        if not future_date_str:
            return jsonify({"error": "La fecha es obligatoria."}), 400

        future_date = datetime.strptime(future_date_str, '%Y-%m-%d')

        last_date = historical_data.index[-1]
        if future_date <= last_date:
            return jsonify({"error": "La fecha debe ser posterior al último dato histórico."}), 400
        days_to_predict = (future_date - last_date).days

        # Genera la predicción iterativa
        last_sequence = scaled_data[-60:] 
        future_predictions = []

        # Realiza la predicción
        input_sequence = last_sequence[-60:].reshape(1, 60, 1)
        predicted_value = model.predict(input_sequence)
        
        # Desescalar la predicción utilizando el scaler cargado
        predicted_value_original = scaler.inverse_transform(predicted_value)
        future_predictions.append(predicted_value_original[0, 0])

        # Preparar las fechas y valores predichos
        prediction_dates = [last_date + timedelta(days=i + 1) for i in range(days_to_predict)]
        predictions = [{"date": date.strftime('%Y-%m-%d'), "predicted_close": float(value)} for date, value in zip(prediction_dates, future_predictions)]

        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)