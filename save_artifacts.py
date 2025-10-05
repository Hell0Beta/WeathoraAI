"""
This script saves the necessary artifacts after training the model.
Run this AFTER training your model to prepare files for the API.
"""

import pickle
import json

def save_model_artifacts(predictor):
    """
    Save scaler and configuration for the Flask API.
    
    Parameters:
    -----------
    predictor : WeatherPredictor instance
        The trained predictor object
    """
    print("\n" + "="*60)
    print("SAVING MODEL ARTIFACTS FOR API")
    print("="*60)
    
    # 1. Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(predictor.scaler, f)
    print("✓ Saved scaler.pkl")
    
    # 2. Save the configuration
    config = {
        'target_var': predictor.target_var,
        'lookback': predictor.lookback,
        'forecast_horizon': predictor.forecast_horizon,
        'feature_cols': predictor.feature_cols,
        'input_shape': [predictor.lookback, len(predictor.feature_cols)],
        'region_bounds': predictor.region_bounds
    }
    
    with open('model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("✓ Saved model_config.json")
    
    print("\n" + "="*60)
    print("FILES CREATED:")
    print("  1. weather_lstm_model.h5  (already saved)")
    print("  2. scaler.pkl             (just created)")
    print("  3. model_config.json      (just created)")
    print("="*60)
    print("\n✅ All artifacts saved! You can now run the Flask API.\n")


# ============================================================================
# USAGE: Add this to the end of your training script
# ============================================================================

if __name__ == "__main__":
    """
    HOW TO USE:
    
    1. After training your model with the main script, add this:
    
    from save_artifacts import save_model_artifacts
    
    # After predictor.train(...)
    save_model_artifacts(predictor)
    
    OR run this as standalone after importing your trained predictor.
    """
    
    print("\n" + "="*60)
    print("HOW TO USE THIS SCRIPT")
    print("="*60)
    print("""
After training your model, add these lines to your training script:

    from save_artifacts import save_model_artifacts
    
    # Train your model
    predictor = WeatherPredictor(...)
    predictor.load_and_process_data()
    predictor.train(...)
    
    # Save artifacts for API
    save_model_artifacts(predictor)

Or import and run manually:
    
    from your_training_script import predictor
    from save_artifacts import save_model_artifacts
    
    save_model_artifacts(predictor)
    """)
    print("="*60 + "\n")