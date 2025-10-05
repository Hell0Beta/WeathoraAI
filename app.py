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
from flask import send_from_directory
import requests
from datetime import datetime, timedelta
import random

app = Flask(__name__)


# Add these imports at the top of app.py

# Add these helper functions after your existing helper functions



CORS(app)  # Enable CORS for all routes

# ============================================================================
# CONFIGURATION
# ============================================================================
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weather_lstm_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
CONFIG_PATH = os.path.join(BASE_DIR, "model_config.json")


model_path = os.path.join(BASE_DIR, "weather_lstm_model.h5")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
config_path = os.path.join(BASE_DIR, "model_config.json")

# ============================================================================
# LOAD MODEL AND CONFIGURATION
# ============================================================================

print("Loading models and configurations...")

# Load CERES model
try:
    ceres_model = keras.models.load_model('weather_lstm_model.h5')
    with open('scaler.pkl', 'rb') as f:
        ceres_scaler = pickle.load(f)
    with open('model_config.json', 'r') as f:
        ceres_config = json.load(f)
    print(f"✓ CERES model loaded")
except Exception as e:
    print(f"✗ CERES model failed: {e}")
    ceres_model = None
    ceres_scaler = None
    ceres_config = None

# Load NVAPS model
try:
    nvaps_model = keras.models.load_model(os.path.join(BASE_DIR, "nvaps_lstm_model.h5"))
    with open(os.path.join(BASE_DIR, "nvaps_scaler.pkl"), 'rb') as f:
        nvaps_scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, "nvaps_config.json"), 'r') as f:
        nvaps_config = json.load(f)
    with open(os.path.join(BASE_DIR, "nvaps_label_encoder.pkl"), 'rb') as f:
        nvaps_label_encoder = pickle.load(f)
    print(f"✓ NVAPS model loaded")
except Exception as e:
    print(f"✗ NVAPS model failed: {e}")
    nvaps_model = None
    nvaps_scaler = None
    nvaps_config = None
    nvaps_label_encoder = None

# Set defaults for backward compatibility
model = ceres_model
scaler = ceres_scaler
config = ceres_config

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_input_sequence(historical_data, lookback, feature_cols):
    """
    Prepare input sequence from historical data.
    
    Parameters:
    -----------
    historical_data : list of dicts
        Historical weather data points
    lookback : int
        Number of past observations to use
    feature_cols : list
        List of feature column names
    
    Returns:
    --------
    numpy array : Scaled input sequence ready for prediction
    """
    # Convert to DataFrame
    df = pd.DataFrame(historical_data)
    
    # Ensure we have all required features
    for col in feature_cols:
        if col not in df.columns:
            if col == 'month_sin' or col == 'month_cos':
                continue  # These will be calculated
            else:
                raise ValueError(f"Missing required feature: {col}")
    
    # Add temporal features if date is provided
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Select and order features
    features = df[feature_cols].values
    
    # Check if we have enough data
    if len(features) < lookback:
        raise ValueError(f"Need at least {lookback} data points, got {len(features)}")
    
    # Take the last 'lookback' observations
    sequence = features[-lookback:]
    
    # Scale the sequence
    sequence_scaled = scaler.transform(sequence)
    
    # Reshape for model input: (1, lookback, n_features)
    sequence_reshaped = sequence_scaled.reshape(1, lookback, len(feature_cols))
    
    return sequence_reshaped

def predict_multiple_steps(initial_sequence, steps, target_var_idx, feature_cols):
    """
    Predict multiple time steps into the future.
    
    Parameters:
    -----------
    initial_sequence : numpy array
        Initial sequence of shape (1, lookback, n_features)
    steps : int
        Number of steps to predict ahead
    target_var_idx : int
        Index of target variable in feature columns
    feature_cols : list
        List of feature column names
    
    Returns:
    --------
    list : Predicted values (original scale)
    """
    predictions = []
    current_sequence = initial_sequence[0].copy()  # Shape: (lookback, n_features)
    
    for _ in range(steps):
        # Reshape for prediction
        X_pred = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        
        # Predict next value (scaled)
        pred_scaled = model.predict(X_pred, verbose=0)[0, 0]
        
        # Store prediction
        predictions.append(pred_scaled)
        
        # Update sequence for next prediction
        new_row = current_sequence[-1].copy()
        new_row[target_var_idx] = pred_scaled
        
        # Slide window
        current_sequence = np.vstack([current_sequence[1:], new_row])
    
    # Inverse transform predictions
    dummy_array = np.zeros((len(predictions), len(feature_cols)))
    dummy_array[:, target_var_idx] = predictions
    predictions_original = scaler.inverse_transform(dummy_array)[:, target_var_idx]
    
    return predictions_original.tolist()

def tpw_to_weather_estimates(tpw_value, month=None, region_temp_avg=27):
    """
    Derive weather parameter estimates from TPW value.
    This is an approximation based on atmospheric physics relationships.
    
    Parameters:
    -----------
    tpw_value : float
        Total Precipitable Water in mm
    month : int
        Month of year (1-12) for seasonal adjustments
    region_temp_avg : float
        Average temperature for the region (default 27°C for Ghana)
    
    Returns:
    --------
    dict : Estimated weather parameters
    """
    # TPW correlations (empirically derived for tropical regions)
    
    # Humidity: Strong positive correlation with TPW
    # TPW range ~20-70mm maps to ~40-95% humidity
    humidity = min(95, max(40, 40 + (tpw_value - 20) * 1.1))
    
    # Cloud cover: Higher TPW = more clouds
    # Non-linear relationship
    if tpw_value < 30:
        cloudcover = 10 + (tpw_value - 20) * 2
    elif tpw_value < 50:
        cloudcover = 30 + (tpw_value - 30) * 2
    else:
        cloudcover = 70 + min(30, (tpw_value - 50) * 0.5)
    cloudcover = max(0, min(100, cloudcover))
    
    # Precipitation: Higher TPW increases rain probability
    if tpw_value < 35:
        precip = 0
    elif tpw_value < 50:
        precip = random.uniform(0, 2)
    elif tpw_value < 60:
        precip = random.uniform(1, 8)
    else:
        precip = random.uniform(5, 20)
    
    # Pressure: Inverse relationship with humidity/TPW
    # Standard pressure ~1013 hPa, varies with moisture
    pressure = 1013 - (humidity - 50) * 0.15
    
    # Temperature adjustment based on moisture
    # More moisture = slightly higher feels-like temp
    temp = region_temp_avg + (humidity - 70) * 0.1
    feelslike = temp + (humidity - 60) * 0.15
    
    # Wind: Generally higher with unstable moist air
    if tpw_value < 40:
        wind_speed = random.uniform(5, 15)
    elif tpw_value < 55:
        wind_speed = random.uniform(10, 25)
    else:
        wind_speed = random.uniform(15, 35)
    
    wind_degree = random.randint(0, 359)
    wind_dirs = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    wind_dir = wind_dirs[int((wind_degree + 22.5) / 45) % 8]
    
    # UV index: Affected by cloud cover
    uv_base = 7 if month and month in [3,4,5,6,7,8] else 5  # Higher in dry season
    uv_index = max(0, uv_base * (1 - cloudcover/150))
    
    return {
        'tpw_mm': round(tpw_value, 2),
        'humidity': round(humidity, 0),
        'cloudcover': round(cloudcover, 0),
        'precip_mm': round(precip, 1),
        'pressure_mb': round(pressure, 0),
        'temperature_c': round(temp, 1),
        'feelslike_c': round(feelslike, 1),
        'wind_speed_kmh': round(wind_speed, 1),
        'wind_degree': wind_degree,
        'wind_dir': wind_dir,
        'uv_index': round(uv_index, 1),
        'estimation_note': 'Derived from TPW model predictions'
    }

def fetch_current_weather(lat=7.9465, lon=-1.0232, api_key=None):
    """
    Fetch current weather from external API (WeatherAPI or OpenWeatherMap).
    Default coordinates are for Kumasi, Ghana.
    
    Parameters:
    -----------
    lat : float
        Latitude
    lon : float
        Longitude
    api_key : str
        API key for weather service
    
    Returns:
    --------
    dict : Current weather data
    """
    if not api_key:
        # Return mock data if no API key
        return {
            'source': 'mock',
            'location': f'Ghana ({lat:.2f}, {lon:.2f})',
            'temperature_c': 28.5,
            'feelslike_c': 32.1,
            'humidity': 78,
            'cloudcover': 40,
            'pressure_mb': 1011,
            'precip_mm': 0,
            'wind_speed_kmh': 15.2,
            'wind_degree': 221,
            'wind_dir': 'SW',
            'uv_index': 6,
            'condition': 'Partly cloudy',
            'note': 'Mock data - provide API key for real data'
        }
    
    try:
        # WeatherAPI.com (free tier available)
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={lat},{lon}"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            current = data['current']
            
            return {
                'source': 'weatherapi.com',
                'location': data['location']['name'],
                'temperature_c': current['temp_c'],
                'feelslike_c': current['feelslike_c'],
                'humidity': current['humidity'],
                'cloudcover': current['cloud'],
                'pressure_mb': current['pressure_mb'],
                'precip_mm': current['precip_mm'],
                'wind_speed_kmh': current['wind_kph'],
                'wind_degree': current['wind_degree'],
                'wind_dir': current['wind_dir'],
                'uv_index': current['uv'],
                'condition': current['condition']['text']
            }
    except Exception as e:
        return {'error': str(e), 'source': 'api_error'}

def prepare_input_sequence_nvaps(historical_data, lookback, feature_cols, scaler_obj):
    """Prepare input for NVAPS model with rolling features."""
    df = pd.DataFrame(historical_data)
    
    # Add temporal features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Add rolling features (CRITICAL - model expects these!)
    for window in [7, 14, 30]:
        df[f'tpw_roll_mean_{window}d'] = df['tpw_mean'].rolling(window, min_periods=1).mean()
        df[f'tpw_roll_std_{window}d'] = df['tpw_mean'].rolling(window, min_periods=1).std()
        df[f'tpw_trend_{window}d'] = df['tpw_mean'] - df[f'tpw_roll_mean_{window}d']
    
    # Fill NaN values from rolling stats
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    # Select features and take last lookback days
    features = df[feature_cols].values[-lookback:]
    
    if len(features) < lookback:
        raise ValueError(f"Need at least {lookback} days of data, got {len(features)}")
    
    features_scaled = scaler_obj.transform(features)
    return features_scaled.reshape(1, lookback, len(feature_cols))

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/', methods=['GET'])
def home():
    """API home endpoint with documentation."""
    return jsonify({
        'message': 'Weather Prediction API',
        'version': '2.0',
        'status': 'online' if model is not None else 'model not loaded',
        'endpoints': {
            '/': 'API documentation (this page)',
            '/health': 'Health check',
            '/predict': 'TPW predictions (POST)',
            '/predict/multi': 'Multi-step TPW predictions (POST)',
            '/model/info': 'Model information (GET)',
            '/weather/predict': 'Weather forecast from TPW (POST) - NEW',
            '/weather/current': 'Current weather + forecast (GET/POST) - NEW',
            '/hihi/': 'jojo'

        },
        'weather_parameters': [
            'temperature_c', 'feelslike_c', 'humidity', 'cloudcover',
            'pressure_mb', 'precip_mm', 'wind_speed_kmh', 'wind_degree',
            'wind_dir', 'uv_index'
        ],
        'documentation': 'Use /weather/predict or /weather/current for weather data'
    })


@app.route('/weather/predict', methods=['POST'])
def weather_predict():
    if nvaps_model is None:
        return jsonify({'error': 'NVAPS model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'historical_data' not in data:
            return jsonify({'error': 'Missing historical_data'}), 400
        
        historical_data = data['historical_data']
        
        # Use nvaps config
        lookback = nvaps_config['lstm_config']['lookback']
        feature_cols = nvaps_config['lstm_config']['feature_cols']
        target_var = nvaps_config['lstm_config']['target_var']
        
        # Prepare input with rolling features
        input_sequence = prepare_input_sequence_nvaps(historical_data, lookback, feature_cols, nvaps_scaler)
        
        # Predict
        pred_scaled = nvaps_model.predict(input_sequence, verbose=0)[0, 0]
        
        # Inverse transform
        target_idx = feature_cols.index(target_var)
        dummy_array = np.zeros((1, len(feature_cols)))
        dummy_array[0, target_idx] = pred_scaled
        tpw_prediction = nvaps_scaler.inverse_transform(dummy_array)[0, target_idx]
        
        # Convert to weather
        forecast_days = data.get('forecast_days', 7)
        last_date = pd.to_datetime(historical_data[-1].get('date', datetime.now()))
        forecast_date = last_date + timedelta(days=forecast_days)
        
        weather_estimates = tpw_to_weather_estimates(tpw_prediction, month=forecast_date.month)
        
        return jsonify({
            'forecast_date': forecast_date.strftime('%Y-%m-%d'),
            'location': data.get('location', {'name': 'Ghana'}),
            'weather': weather_estimates,
            'tpw_data': {'predicted_tpw_mm': round(tpw_prediction, 2)}
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/weather/current', methods=['GET', 'POST'])
def weather_current():
    """
    Get current weather conditions, optionally enhanced with TPW prediction.
    
    Query params or JSON:
    {
        "lat": 7.9465,           // optional
        "lon": -1.0232,          // optional
        "api_key": "your_key",   // optional, for real weather data
        "include_forecast": true // optional, adds TPW-based forecast
    }
    
    Returns current weather with optional TPW-enhanced forecast.
    """
    try:
        # Get parameters
        if request.method == 'POST':
            data = request.get_json() or {}
        else:
            data = {}
        
        lat = float(data.get('lat', request.args.get('lat', 7.9465)))
        lon = float(data.get('lon', request.args.get('lon', -1.0232)))
        api_key = data.get('api_key', request.args.get('api_key'))
        include_forecast = data.get('include_forecast', 
                                   request.args.get('include_forecast', 'false').lower() == 'true')
        
        # Fetch current weather
        current_weather = fetch_current_weather(lat, lon, api_key)
        
        response = {
            'current': current_weather,
            'location': {
                'lat': lat,
                'lon': lon
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add TPW-based forecast if requested and model is loaded
        if include_forecast and model is not None:
            try:
                # Use current humidity as proxy to estimate current TPW
                if 'humidity' in current_weather:
                    humidity = current_weather['humidity']
                    estimated_tpw = 20 + (humidity - 40) / 1.1  # Reverse of our formula
                    
                    # Predict future TPW (simplified - would need actual sequence)
                    future_tpw = estimated_tpw * 1.05  # Simple 5% increase assumption
                    
                    forecast_weather = tpw_to_weather_estimates(
                        future_tpw,
                        month=(datetime.now() + timedelta(days=7)).month
                    )
                    
                    response['forecast_7day'] = {
                        'date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d'),
                        'weather': {
                            'temperature_c': forecast_weather['temperature_c'],
                            'feelslike_c': forecast_weather['feelslike_c'],
                            'humidity': int(forecast_weather['humidity']),
                            'cloudcover': int(forecast_weather['cloudcover']),
                            'pressure_mb': int(forecast_weather['pressure_mb']),
                            'precip_mm': forecast_weather['precip_mm'],
                            'wind_speed_kmh': forecast_weather['wind_speed_kmh'],
                            'wind_degree': forecast_weather['wind_degree'],
                            'wind_dir': forecast_weather['wind_dir'],
                            'uv_index': forecast_weather['uv_index']
                        },
                        'note': 'TPW-based forecast estimate'
                    }
            except Exception as e:
                response['forecast_error'] = str(e)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/dashboard')
def dashboard():
    return send_from_directory('.', 'dashboard.html')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'config_loaded': config is not None,
        'timestamp': datetime.now().isoformat(),
        'koko':'kko'
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information and configuration."""
    if config is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'LSTM Neural Network',
        'target_variable': config.get('target_var'),
        'lookback_period': config.get('lookback'),
        'forecast_horizon': config.get('forecast_horizon'),
        'feature_columns': config.get('feature_cols'),
        'input_shape': config.get('input_shape'),
        'model_path': MODEL_PATH
    })

@app.route('/predict/auto', methods=['POST'])
def predict_auto():
    """
    Make prediction using automatically fetched latest data from NetCDF file.
    
    Expected JSON format:
    {
        "nc_file_path": "path/to/your/file.nc",  // optional, uses default if not provided
        "months_ahead": 3  // optional, defaults to model's forecast_horizon
    }
    
    Returns prediction using the most recent data from the file.
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        import xarray as xr
        
        data = request.get_json() or {}
        nc_file_path = data.get('nc_file_path', 'CERES_EBAF-TOA_Edition4.2_200003-202407.nc')
        
        # Load NetCDF file
        try:
            ds = xr.open_dataset(nc_file_path, decode_cf=False, engine='netcdf4')
        except ValueError:
            try:
                ds = xr.open_dataset(nc_file_path, decode_cf=False, engine='h5netcdf')
            except (ValueError, ImportError):
                ds = xr.open_dataset(nc_file_path, decode_cf=False, engine='scipy')
        
        # Get region bounds from config
        region_bounds = config.get('region_bounds', {'lat': (4, 12), 'lon': (-4, 2)})
        lookback = config['lookback']
        feature_cols = config['feature_cols']
        target_var = config['target_var']
        
        # Extract latest data
        historical_data = []
        
        for var in ['toa_net_all_mon', 'toa_sw_all_mon', 'toa_lw_all_mon']:
            if var in ds.variables:
                var_data = ds[var]
                lat_slice = slice(region_bounds['lat'][0], region_bounds['lat'][1])
                lon_slice = slice(region_bounds['lon'][0], region_bounds['lon'][1])
                var_regional = var_data.sel(lat=lat_slice, lon=lon_slice)
                var_timeseries = var_regional.mean(dim=['lat', 'lon'])
                
                if 'scale_factor' in var_data.attrs:
                    var_timeseries = var_timeseries * var_data.attrs['scale_factor']
                if 'add_offset' in var_data.attrs:
                    var_timeseries = var_timeseries + var_data.attrs['add_offset']
                
                # Take last 'lookback' months
                values = var_timeseries.values[-lookback:]
                
                for i, val in enumerate(values):
                    if i >= len(historical_data):
                        historical_data.append({})
                    historical_data[i][var] = float(val)
        
        # Add time information
        time_values = ds['time'].values[-lookback:]
        time_units = ds['time'].attrs.get('units', 'days since 2000-03-01 00:00:00')
        
        if 'since' in time_units:
            units_split = time_units.split('since')
            origin = pd.to_datetime(units_split[1].strip())
            
            for i, t in enumerate(time_values):
                if 'days' in units_split[0]:
                    date = origin + timedelta(days=float(t))
                else:
                    date = origin + timedelta(hours=float(t))
                
                historical_data[i]['date'] = date.strftime('%Y-%m-%d')
                historical_data[i]['month'] = date.month
        
        ds.close()
        
        # Now make prediction with this auto-fetched data
        input_sequence = prepare_input_sequence(historical_data, lookback, feature_cols)
        pred_scaled = model.predict(input_sequence, verbose=0)[0, 0]
        
        target_idx = feature_cols.index(target_var)
        dummy_array = np.zeros((1, len(feature_cols)))
        dummy_array[0, target_idx] = pred_scaled
        prediction = scaler.inverse_transform(dummy_array)[0, target_idx]
        
        last_date = pd.to_datetime(historical_data[-1]['date'])
        forecast_horizon = config['forecast_horizon']
        forecast_date = last_date + pd.DateOffset(months=forecast_horizon)
        
        response = {
            'prediction': float(prediction),
            'forecast_date': forecast_date.strftime('%Y-%m-%d'),
            'target_variable': target_var,
            'forecast_horizon_months': forecast_horizon,
            'metadata': {
                'last_data_date': last_date.strftime('%Y-%m-%d'),
                'lookback_months': lookback,
                'data_points_used': len(historical_data),
                'data_source': nc_file_path,
                'auto_fetched': True
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a single prediction.
    
    Expected JSON format:
    {
        "historical_data": [
            {
                "toa_net_all_mon": 100.5,
                "toa_sw_all_mon": 200.3,
                "toa_lw_all_mon": 150.2,
                "month": 1,
                "date": "2024-01-01"  // optional
            },
            ... (need 'lookback' number of data points)
        ]
    }
    
    Returns:
    {
        "prediction": 105.3,
        "confidence": "high",
        "forecast_date": "2024-04-01",
        "metadata": {...}
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get request data
        data = request.get_json()
        
        if 'historical_data' not in data:
            return jsonify({'error': 'Missing historical_data in request'}), 400
        
        historical_data = data['historical_data']
        lookback = config['lookback']
        feature_cols = config['feature_cols']
        target_var = config['target_var']
        forecast_horizon = config['forecast_horizon']
        
        # Validate data length
        if len(historical_data) < lookback:
            return jsonify({
                'error': f'Need at least {lookback} historical data points, got {len(historical_data)}'
            }), 400
        
        # Prepare input sequence
        input_sequence = prepare_input_sequence(historical_data, lookback, feature_cols)
        
        # Make prediction
        pred_scaled = model.predict(input_sequence, verbose=0)[0, 0]
        
        # Inverse transform
        target_idx = feature_cols.index(target_var)
        dummy_array = np.zeros((1, len(feature_cols)))
        dummy_array[0, target_idx] = pred_scaled
        prediction = scaler.inverse_transform(dummy_array)[0, target_idx]
        
        # Calculate forecast date
        last_date = pd.to_datetime(historical_data[-1].get('date', datetime.now()))
        forecast_date = last_date + pd.DateOffset(months=forecast_horizon)
        
        # Prepare response
        response = {
            'prediction': float(prediction),
            'forecast_date': forecast_date.strftime('%Y-%m-%d'),
            'target_variable': target_var,
            'forecast_horizon_months': forecast_horizon,
            'metadata': {
                'last_data_date': last_date.strftime('%Y-%m-%d'),
                'lookback_months': lookback,
                'data_points_used': len(historical_data)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/multi', methods=['POST'])
def predict_multi():
    """
    Make multiple predictions (multi-step forecast).
    
    Expected JSON format:
    {
        "historical_data": [...],
        "steps": 6  // number of months to predict ahead
    }
    
    Returns:
    {
        "predictions": [
            {"date": "2024-04-01", "value": 105.3},
            {"date": "2024-05-01", "value": 108.1},
            ...
        ],
        "metadata": {...}
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get request data
        data = request.get_json()
        
        if 'historical_data' not in data:
            return jsonify({'error': 'Missing historical_data in request'}), 400
        
        historical_data = data['historical_data']
        steps = data.get('steps', 6)  # Default to 6 months
        
        lookback = config['lookback']
        feature_cols = config['feature_cols']
        target_var = config['target_var']
        target_idx = feature_cols.index(target_var)
        
        # Validate
        if len(historical_data) < lookback:
            return jsonify({
                'error': f'Need at least {lookback} historical data points'
            }), 400
        
        if steps < 1 or steps > 24:
            return jsonify({'error': 'Steps must be between 1 and 24'}), 400
        
        # Prepare input sequence
        input_sequence = prepare_input_sequence(historical_data, lookback, feature_cols)
        
        # Make multi-step predictions
        predictions = predict_multiple_steps(input_sequence, steps, target_idx, feature_cols)
        
        # Create forecast dates
        last_date = pd.to_datetime(historical_data[-1].get('date', datetime.now()))
        forecast_dates = [last_date + pd.DateOffset(months=i+1) for i in range(steps)]
        
        # Format response
        predictions_list = [
            {
                'date': date.strftime('%Y-%m-%d'),
                'value': float(pred),
                'month_ahead': i + 1
            }
            for i, (date, pred) in enumerate(zip(forecast_dates, predictions))
        ]
        
        response = {
            'predictions': predictions_list,
            'target_variable': target_var,
            'metadata': {
                'last_data_date': last_date.strftime('%Y-%m-%d'),
                'total_predictions': steps,
                'lookback_months': lookback,
                'data_points_used': len(historical_data)
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Make predictions for multiple sequences at once.
    
    Expected JSON format:
    {
        "sequences": [
            {"historical_data": [...]},
            {"historical_data": [...]},
            ...
        ]
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'sequences' not in data:
            return jsonify({'error': 'Missing sequences in request'}), 400
        
        sequences = data['sequences']
        results = []
        
        for idx, seq_data in enumerate(sequences):
            try:
                historical_data = seq_data['historical_data']
                lookback = config['lookback']
                feature_cols = config['feature_cols']
                target_var = config['target_var']
                target_idx = feature_cols.index(target_var)
                
                # Prepare and predict
                input_sequence = prepare_input_sequence(historical_data, lookback, feature_cols)
                pred_scaled = model.predict(input_sequence, verbose=0)[0, 0]
                
                # Inverse transform
                dummy_array = np.zeros((1, len(feature_cols)))
                dummy_array[0, target_idx] = pred_scaled
                prediction = scaler.inverse_transform(dummy_array)[0, target_idx]
                
                results.append({
                    'sequence_id': idx,
                    'prediction': float(prediction),
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'sequence_id': idx,
                    'error': str(e),
                    'status': 'failed'
                })
        
        return jsonify({
            'results': results,
            'total': len(sequences),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'failed')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌤️  WEATHER PREDICTION API SERVER")
    print("="*60)
    print("\nStarting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /              - API documentation")
    print("  GET  /health        - Health check")
    print("  GET  /model/info    - Model information")
    print("  POST /predict       - Single prediction")
    print("  POST /predict/multi - Multi-step forecast")
    print("  POST /predict/batch - Batch predictions")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)