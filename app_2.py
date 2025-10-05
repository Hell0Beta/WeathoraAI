from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
import json
import xgboost as xgb
import random

app = Flask(__name__)
CORS(app)

# ============================================================================
# LOAD ALL MODELS
# ============================================================================

print("Loading models...")

# Model variables
lstm_model = None
xgb_model = None
scaler = None
label_encoder = None
config = None

# Try to load LSTM model
for path in ['nvaps_lstm_model.h5', 'weather_lstm_model.h5']:
    if os.path.exists(path):
        lstm_model = keras.models.load_model(path)
        print(f"✓ LSTM: {path}")
        break

# Try to load scaler
for path in ['nvaps_scaler.pkl']:
    if os.path.exists(path):
        with open(path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✓ Scaler: {path}")
        break

# Try to load config
for path in ['nvaps_config.json', 'model_config.json']:
    if os.path.exists(path):
        with open(path, 'r') as f:
            config = json.load(f)
        print(f"✓ Config: {path}")
        break

# Try to load XGBoost
if os.path.exists('nvaps_xgboost_model.json'):
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model('nvaps_xgboost_model.json')
    print("✓ XGBoost: nvaps_xgboost_model.json")

# Try to load label encoder
if os.path.exists('nvaps_label_encoder.pkl'):
    with open('nvaps_label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("✓ Label Encoder: nvaps_label_encoder.pkl")

# Check status
if lstm_model and scaler and config:
    print("\n🚀 API Ready!")
else:
    print("\n⚠️ Error: Required models not found")
    print("Run NVAPS_training.py first to generate model files")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_random_tpw_data(days=14):
    """
    Generate random but realistic TPW data for Ghana region.
    TPW typically ranges from 20-70mm in tropical regions.
    """
    data = []
    base_date = datetime.now() - timedelta(days=days)
    
    # Seasonal pattern (higher in rainy season)
    month = base_date.month
    if month in [3, 4, 5, 6, 7, 8, 9, 10]:  # Rainy season
        base_tpw = random.uniform(40, 60)
    else:  # Dry season
        base_tpw = random.uniform(25, 45)
    
    for i in range(days):
        date = base_date + timedelta(days=i)
        
        # Add daily variation
        tpw_mean = base_tpw + random.uniform(-5, 5)
        tpw_mean = max(20, min(70, tpw_mean))  # Clamp to realistic range
        
        data.append({
            'tpw_mean': round(tpw_mean, 2),
            'tpw_std': round(random.uniform(3, 8), 2),
            'tpw_min': round(tpw_mean - random.uniform(5, 10), 2),
            'tpw_max': round(tpw_mean + random.uniform(5, 10), 2),
            'month': date.month,
            'day_of_year': date.timetuple().tm_yday,
            'date': date.strftime('%Y-%m-%d')
        })
    
    return data


def prepare_sequence(data, lookback, feature_cols):
    """Prepare input sequence for LSTM model."""
    df = pd.DataFrame(data)
    
    # Add temporal features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Select available features
    available = [col for col in feature_cols if col in df.columns]
    features = df[available].values[-lookback:]
    
    if len(features) < lookback:
        raise ValueError(f"Need {lookback} data points, got {len(features)}")
    
    scaled = scaler.transform(features)
    return scaled.reshape(1, lookback, len(available))


def tpw_to_weather(tpw_value, month):
    """Convert TPW to weather parameters."""
    # Humidity (40-95%)
    humidity = min(95, max(40, 40 + (tpw_value - 20) * 1.1))
    
    # Cloud cover (0-100%)
    if tpw_value < 30:
        cloudcover = 10 + (tpw_value - 20) * 2
    elif tpw_value < 50:
        cloudcover = 30 + (tpw_value - 30) * 2
    else:
        cloudcover = 70 + min(30, (tpw_value - 50) * 0.5)
    cloudcover = max(0, min(100, cloudcover))
    
    # Precipitation (0-20mm)
    if tpw_value < 35:
        precip = 0
    elif tpw_value < 50:
        precip = random.uniform(0, 2)
    elif tpw_value < 60:
        precip = random.uniform(1, 8)
    else:
        precip = random.uniform(5, 20)
    
    # Pressure (990-1020 mb)
    pressure = 1013 - (humidity - 50) * 0.15
    
    # Temperature (20-35°C for Ghana)
    temp = 27 + (humidity - 70) * 0.1
    feelslike = temp + (humidity - 60) * 0.15
    
    # Wind (5-35 km/h)
    if tpw_value < 40:
        wind_speed = random.uniform(5, 15)
    elif tpw_value < 55:
        wind_speed = random.uniform(10, 25)
    else:
        wind_speed = random.uniform(15, 35)
    
    wind_degree = random.randint(0, 359)
    wind_dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    wind_dir = wind_dirs[int((wind_degree + 22.5) / 45) % 8]
    
    # UV Index (0-11)
    uv_base = 7 if month in [3,4,5,6,7,8] else 5
    uv_index = max(0, uv_base * (1 - cloudcover/150))
    
    return {
        'temperature_c': round(temp, 1),
        'feelslike_c': round(feelslike, 1),
        'humidity': int(round(humidity)),
        'cloudcover': int(round(cloudcover)),
        'pressure_mb': int(round(pressure)),
        'precip_mm': round(precip, 1),
        'wind_speed_kmh': round(wind_speed, 1),
        'wind_degree': wind_degree,
        'wind_dir': wind_dir,
        'uv_index': round(uv_index, 1)
    }


def predict_weather_condition(data, lookback, feature_cols):
    """Predict weather condition using XGBoost if available."""
    if not xgb_model or not label_encoder:
        return None
    
    try:
        df = pd.DataFrame(data)
        
        # Add temporal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Build feature sequence
        features = []
        for i in range(lookback):
            idx = -(lookback - i)
            for col in feature_cols:
                if col in df.columns:
                    features.append(df[col].iloc[idx])
        
        X = np.array(features).reshape(1, -1)
        proba = xgb_model.predict_proba(X)[0]
        predicted_class = np.argmax(proba)
        
        return {
            'condition': label_encoder.classes_[predicted_class],
            'confidence': float(proba[predicted_class]),
            'probabilities': {
                label_encoder.classes_[i]: float(proba[i]) 
                for i in range(len(proba))
            }
        }
    except Exception as e:
        return None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/predict/random', methods=['GET', 'POST'])
def predict_random():
    """
    Predict weather using randomly generated TPW data.
    
    Optional JSON body:
    {
        "days_history": 14,      // optional, default 14
        "forecast_days": 7       // optional, default 7
    }
    """
    if not lstm_model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json() if request.method == 'POST' else {}
        
        days_history = data.get('days_history', 14)
        forecast_days = data.get('forecast_days', 7)
        
        # Validate inputs
        days_history = max(7, min(60, days_history))
        forecast_days = max(1, min(30, forecast_days))
        
        # Get config
        lstm_cfg = config.get('lstm_config', {})
        lookback = lstm_cfg.get('lookback', 14)
        feature_cols = lstm_cfg.get('feature_cols', [])
        target_var = lstm_cfg.get('target_var', 'tpw_mean')
        
        # Generate random data
        historical_data = generate_random_tpw_data(max(lookback, days_history))
        
        # Prepare sequence
        input_seq = prepare_sequence(historical_data, lookback, feature_cols)
        
        # Predict TPW
        pred_scaled = lstm_model.predict(input_seq, verbose=0)[0, 0]
        
        target_idx = feature_cols.index(target_var)
        dummy = np.zeros((1, len(feature_cols)))
        dummy[0, target_idx] = pred_scaled
        tpw_prediction = scaler.inverse_transform(dummy)[0, target_idx]
        
        # Get forecast date
        last_date = pd.to_datetime(historical_data[-1]['date'])
        forecast_date = last_date + timedelta(days=forecast_days)
        
        # Convert to weather parameters
        weather = tpw_to_weather(tpw_prediction, forecast_date.month)
        
        # Try to get weather condition classification
        condition = predict_weather_condition(
            historical_data, 
            config.get('xgb_config', {}).get('lookback', 14),
            config.get('xgb_config', {}).get('feature_cols', [])
        )
        
        response = {
            'forecast_date': forecast_date.strftime('%Y-%m-%d'),
            'weather': weather,
            'tpw_predicted_mm': round(tpw_prediction, 2),
            'metadata': {
                'data_source': 'randomly_generated',
                'last_data_date': last_date.strftime('%Y-%m-%d'),
                'days_history_used': len(historical_data),
                'forecast_days_ahead': forecast_days,
                'model': 'LSTM + TPW conversion'
            }
        }
        
        if condition:
            response['weather_condition'] = condition
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/upload', methods=['POST'])
def predict_upload():
    """
    Predict weather using uploaded historical TPW data.
    
    Required JSON body:
    {
        "historical_data": [
            {
                "tpw_mean": 45.2,
                "tpw_std": 5.1,
                "tpw_min": 35.0,
                "tpw_max": 55.0,
                "month": 10,
                "day_of_year": 277,
                "date": "2025-10-04"
            },
            // ... need at least 14 data points (or your model's lookback)
        ],
        "forecast_days": 7  // optional, default 7
    }
    """
    if not lstm_model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data or 'historical_data' not in data:
            return jsonify({
                'error': 'Missing historical_data',
                'required_format': {
                    'historical_data': [
                        {
                            'tpw_mean': 'float',
                            'tpw_std': 'float',
                            'tpw_min': 'float',
                            'tpw_max': 'float',
                            'month': 'int (1-12)',
                            'day_of_year': 'int (1-365)',
                            'date': 'string (YYYY-MM-DD)'
                        }
                    ]
                }
            }), 400
        
        historical_data = data['historical_data']
        forecast_days = data.get('forecast_days', 7)
        forecast_days = max(1, min(30, forecast_days))
        
        # Get config
        lstm_cfg = config.get('lstm_config', {})
        lookback = lstm_cfg.get('lookback', 14)
        feature_cols = lstm_cfg.get('feature_cols', [])
        target_var = lstm_cfg.get('target_var', 'tpw_mean')
        
        # Validate data
        if len(historical_data) < lookback:
            return jsonify({
                'error': f'Need at least {lookback} data points, got {len(historical_data)}'
            }), 400
        
        # Prepare sequence
        input_seq = prepare_sequence(historical_data, lookback, feature_cols)
        
        # Predict TPW
        pred_scaled = lstm_model.predict(input_seq, verbose=0)[0, 0]
        
        target_idx = feature_cols.index(target_var)
        dummy = np.zeros((1, len(feature_cols)))
        dummy[0, target_idx] = pred_scaled
        tpw_prediction = scaler.inverse_transform(dummy)[0, target_idx]
        
        # Get forecast date
        last_date = pd.to_datetime(historical_data[-1]['date'])
        forecast_date = last_date + timedelta(days=forecast_days)
        
        # Convert to weather parameters
        weather = tpw_to_weather(tpw_prediction, forecast_date.month)
        
        # Try to get weather condition classification
        condition = predict_weather_condition(
            historical_data,
            config.get('xgb_config', {}).get('lookback', 14),
            config.get('xgb_config', {}).get('feature_cols', [])
        )
        
        response = {
            'forecast_date': forecast_date.strftime('%Y-%m-%d'),
            'weather': weather,
            'tpw_predicted_mm': round(tpw_prediction, 2),
            'metadata': {
                'data_source': 'user_uploaded',
                'last_data_date': last_date.strftime('%Y-%m-%d'),
                'data_points_used': len(historical_data),
                'forecast_days_ahead': forecast_days,
                'model': 'LSTM + TPW conversion'
            }
        }
        
        if condition:
            response['weather_condition'] = condition
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Check API health and model status."""
    return jsonify({
        'status': 'healthy' if lstm_model else 'degraded',
        'models_loaded': {
            'lstm': lstm_model is not None,
            'xgboost': xgb_model is not None,
            'scaler': scaler is not None,
            'label_encoder': label_encoder is not None,
            'config': config is not None
        },
        'endpoints': {
            '/predict/random': 'Generate prediction with random data (GET/POST)',
            '/predict/upload': 'Predict using uploaded historical data (POST)',
            '/health': 'This endpoint (GET)'
        },
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("WEATHER PREDICTION API")
    print("="*60)
    print("\nEndpoints:")
    print("  GET/POST /predict/random  - Random data prediction")
    print("  POST     /predict/upload  - Upload your own data")
    print("  GET      /health          - Health check")
    print("\nServer starting at http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)