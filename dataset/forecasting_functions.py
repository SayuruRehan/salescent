
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime

def load_models_and_data():
    """Load all saved models and data"""

    # Load models
    base_model = joblib.load('models/xgboost_base_model.pkl')
    enhanced_model = joblib.load('models/xgboost_enhanced_model.pkl')

    # Load artifacts
    with open('models/model_artifacts.pkl', 'rb') as f:
        artifacts = pickle.load(f)

    # Load data
    master_df = pd.read_pickle('data/master_df.pkl')
    enhanced_catalog = pd.read_pickle('data/enhanced_catalog.pkl')
    model_data = pd.read_pickle('data/model_data.pkl')

    return {
        'base_model': base_model,
        'enhanced_model': enhanced_model,
        'artifacts': artifacts,
        'master_df': master_df,
        'enhanced_catalog': enhanced_catalog,
        'model_data': model_data
    }

def generate_product_forecast(start_date, end_date, data_dict):
    """Generate product-level forecasts (Clue 2)"""

    model = data_dict['base_model']
    model_data = data_dict['model_data']
    base_features = data_dict['artifacts']['base_features']
    seasonal_factors = data_dict['artifacts']['seasonal_factors']

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
        # Generate base predictions
        forecast_features = active_products[base_features]
        predictions = model.predict(forecast_features)
        predictions = np.maximum(predictions, 0)

        # Apply seasonal adjustments
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        target_month = start_dt.month
        target_season = get_season(target_month)

        adjusted_predictions = []
        for idx, (_, row) in enumerate(active_products.iterrows()):
            base_pred = predictions[idx]
            category = row['category']

            if target_season in seasonal_factors and category in seasonal_factors[target_season]:
                factor = seasonal_factors[target_season][category]
                adjusted_pred = base_pred * factor
            else:
                adjusted_pred = base_pred

            adjusted_predictions.append(max(0, round(adjusted_pred)))

        # Create forecast list
        forecast_list = []
        for idx, (_, row) in enumerate(active_products.iterrows()):
            forecast_list.append({
                "productId": row['productId'],
                "category": row['category'],
                "forecastedQuantity": int(adjusted_predictions[idx])
            })

        return {
            "periodStart": start_date,
            "periodEnd": end_date,
            "forecast": sorted(forecast_list, key=lambda x: x['forecastedQuantity'], reverse=True)
        }

    return {"error": "No active products found"}

def generate_attribute_forecast(start_date, end_date, data_dict):
    """Generate attribute-level forecasts (Clue 3)"""

    model = data_dict['base_model']
    model_data = data_dict['model_data']
    enhanced_catalog = data_dict['enhanced_catalog']
    base_features = data_dict['artifacts']['base_features']

    # Get latest data and merge with enhanced attributes
    latest_month = model_data['year_month'].max()
    latest_data = model_data[model_data['year_month'] == latest_month].copy()

    # Merge with enhanced attributes
    latest_with_attrs = latest_data.merge(
        enhanced_catalog[['productId', 'enhanced_color', 'product_season', 'material', 'style']],
        on='productId',
        how='left'
    )

    # Filter active products
    active_products = latest_with_attrs[
        (latest_with_attrs['monthly_sales'] > 0) | 
        (latest_with_attrs['sales_3month_ma'] > 0) |
        (latest_with_attrs['transaction_count'] > 0)
    ].copy()

    if len(active_products) > 0:
        # Generate predictions
        forecast_features = active_products[base_features]
        predictions = model.predict(forecast_features)
        predictions = np.maximum(predictions, 0)

        active_products = active_products.copy()
        active_products['xgb_forecast'] = predictions

        # Aggregate by attributes
        attribute_forecasts = active_products.groupby(['category', 'enhanced_color', 'product_season'])['xgb_forecast'].sum().reset_index()
        attribute_forecasts.columns = ['category', 'color', 'season', 'forecastedQuantity']
        attribute_forecasts['forecastedQuantity'] = attribute_forecasts['forecastedQuantity'].round().astype(int)

        # Filter significant forecasts
        significant_forecasts = attribute_forecasts[
            (attribute_forecasts['color'] != 'neutral') &
            (attribute_forecasts['forecastedQuantity'] >= 10)
        ].sort_values('forecastedQuantity', ascending=False)

        forecast_list = []
        for _, row in significant_forecasts.head(30).iterrows():
            forecast_list.append({
                "category": row['category'],
                "color": row['color'],
                "season": row['season'],
                "forecastedQuantity": int(row['forecastedQuantity'])
            })

        return {
            "periodStart": start_date,
            "periodEnd": end_date,
            "forecast": forecast_list
        }

    return {"error": "No active products found"}

def get_season(month):
    """Get season from month number"""
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'
