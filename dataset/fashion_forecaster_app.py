# Save this as: simple_forecasting.py

import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime

def load_models_simple():
    """Simplified model loading with error handling"""
    try:
        print("Loading models...")
        
        # Load models
        base_model = joblib.load('models/xgboost_base_model.pkl')
        
        # Load artifacts
        with open('models/model_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
        
        # Load data
        master_df = pd.read_pickle('data/master_df.pkl')
        model_data = pd.read_pickle('data/model_data.pkl')
        enhanced_catalog = pd.read_pickle('data/enhanced_catalog.pkl')
        
        return {
            'base_model': base_model,
            'artifacts': artifacts,
            'master_df': master_df,
            'model_data': model_data,
            'enhanced_catalog': enhanced_catalog
        }
        
    except Exception as e:
        print(f"Error loading: {e}")
        return None

def simple_product_forecast(start_date, end_date):
    """Simple forecasting without complex dependencies"""
    
    data = load_models_simple()
    if data is None:
        return {"error": "Could not load models"}
    
    model = data['base_model']
    model_data = data['model_data']
    base_features = data['artifacts']['base_features']
    
    # Get latest data
    latest_month = model_data['year_month'].max()
    latest_data = model_data[model_data['year_month'] == latest_month].copy()
    
    # Filter active products
    active_products = latest_data[
        (latest_data['monthly_sales'] > 0) | 
        (latest_data['sales_3month_ma'] > 0) |
        (latest_data['transaction_count'] > 0)
    ].copy()
    
    if len(active_products) > 0:
        # Make predictions
        forecast_features = active_products[base_features]
        predictions = model.predict(forecast_features)
        predictions = np.maximum(predictions, 0)
        
        # Create results
        results = []
        for idx, (_, row) in enumerate(active_products.iterrows()):
            results.append({
                "productId": row['productId'],
                "category": row['category'],
                "forecastedQuantity": int(round(predictions[idx]))
            })
        
        return {
            "periodStart": start_date,
            "periodEnd": end_date,
            "forecast": sorted(results, key=lambda x: x['forecastedQuantity'], reverse=True)
        }
    
    return {"error": "No active products found"}

# Test the simple version
if __name__ == "__main__":
    print("Testing simple forecasting...")
    result = simple_product_forecast("2025-06-01", "2025-06-30")
    
    if "error" not in result:
        print(f"✅ Success! Generated {len(result['forecast'])} forecasts")
        print("Top 5:")
        for i, item in enumerate(result['forecast'][:5], 1):
            print(f"  {i}. {item['productId']}: {item['forecastedQuantity']} units")
    else:
        print(f"❌ Error: {result['error']}")