import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Set page config
st.set_page_config(
    page_title="🌡️ Weather Prediction",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 12px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# OPTIONAL TENSORFLOW IMPORT (Safe for deployment)
lstm_model = None
tensorflow_available = False
try:
    import tensorflow as tf
    tensorflow_available = True
except ImportError:
    tensorflow_available = False
    st.warning("⚠️ TensorFlow not available. LSTM predictions will be skipped on deployment...")

# ============================================================================
# 1. LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """Load pre-trained models from pickle files"""
    models = {
        'rf_model': None,
        'xgb_model': None,
        'lstm_model': None,
        'scaler': None
    }
    
    try:
        # Random Forest
        with open('models/random_forest.pkl', 'rb') as f:
            models['rf_model'] = pickle.load(f)
        st.sidebar.success("✅ Random Forest loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Random Forest: {str(e)}")
    
    try:
        # XGBoost
        with open('models/xgboost.pkl', 'rb') as f:
            models['xgb_model'] = pickle.load(f)
        st.sidebar.success("✅ XGBoost loaded")
    except Exception as e:
        st.sidebar.error(f"❌ XGBoost: {str(e)}")
    
    # LSTM is optional - only load if TensorFlow available
    if tensorflow_available:
        try:
            lstm_model_temp = tf.keras.models.load_model('models/lstm.h5')
            models['lstm_model'] = lstm_model_temp
            st.sidebar.success("✅ LSTM loaded")
        except Exception as e:
            st.sidebar.warning(f"⚠️ LSTM could not load: {str(e)}")
            models['lstm_model'] = None
    else:
        models['lstm_model'] = None
    
    try:
        with open('models/scaler.pkl', 'rb') as f:
            models['scaler'] = pickle.load(f)
    except Exception as e:
        st.sidebar.warning(f"⚠️ Scaler: {str(e)}")
    
    return models

# Load all models
rf_model = None
xgb_model = None
lstm_model = None
scaler = None

try:
    models_dict = load_models()
    rf_model = models_dict['rf_model']
    xgb_model = models_dict['xgb_model']
    lstm_model = models_dict['lstm_model']
    scaler = models_dict['scaler']
except Exception as e:
    st.error(f"Error loading models: {str(e)}")

# ============================================================================
# 2. FETCH WEATHER DATA
# ============================================================================

def get_weather_data(lat, lon):
    """Fetch real weather data from OpenWeatherMap API"""
    try:
        # Using Open-Meteo (free, no API key needed)
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,pressure_msl&timezone=auto"
        
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        current = data['current']
        return {
            'temperature': current['temperature_2m'],
            'humidity': current['relative_humidity_2m'],
            'wind_speed': current['wind_speed_10m'],
            'pressure': current['pressure_msl']
        }
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

def get_coordinates(city_name):
    """Get latitude and longitude from city name"""
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if data.get('results'):
            result = data['results'][0]
            return result['latitude'], result['longitude'], result.get('admin1', '')
        else:
            st.error(f"City '{city_name}' not found")
            return None, None, None
    except Exception as e:
        st.error(f"Error getting coordinates: {str(e)}")
        return None, None, None

# ============================================================================
# 3. PREPARE FEATURES FOR PREDICTION
# ============================================================================

def prepare_features(temp, humidity, wind_speed, pressure):
    """Prepare features for model prediction"""
    features = np.array([[temp, humidity, wind_speed, pressure]])
    
    # Scale features if scaler is available
    if scaler is not None:
        try:
            features = scaler.transform(features)
        except Exception as e:
            st.warning(f"Scaler warning: {str(e)}")
    
    return features

# ============================================================================
# 4. MAKE PREDICTIONS
# ============================================================================

def make_predictions(features):
    """Make predictions using loaded models"""
    predictions = {
        'RF': None,
        'XGB': None,
        'LSTM': None,
        'Ensemble': None
    }
    
    # Random Forest prediction
    if rf_model is not None:
        try:
            predictions['RF'] = float(rf_model.predict(features)[0])
        except Exception as e:
            st.warning(f"RF prediction error: {str(e)}")
    
    # XGBoost prediction
    if xgb_model is not None:
        try:
            predictions['XGB'] = float(xgb_model.predict(features)[0])
        except Exception as e:
            st.warning(f"XGB prediction error: {str(e)}")
    
    # LSTM prediction (optional - only if TensorFlow available and model loaded)
    if lstm_model is not None and tensorflow_available:
        try:
            # Reshape for LSTM: (batch_size, time_steps, features)
            lstm_features = features.reshape((features.shape[0], 1, features.shape[1]))
            predictions['LSTM'] = float(lstm_model.predict(lstm_features, verbose=0)[0][0])
        except Exception as e:
            st.warning(f"LSTM prediction warning: {str(e)}")
            predictions['LSTM'] = None
    
    # Ensemble average (only for available predictions)
    available_preds = [v for v in predictions.values() if v is not None]
    if available_preds:
        predictions['Ensemble'] = np.mean(available_preds)
    
    return predictions

# ============================================================================
# 5. SIDEBAR - LOCATION INPUT
# ============================================================================

st.sidebar.title("🌍 Weather Prediction System")
st.sidebar.markdown("---")

# Model status
st.sidebar.subheader("📊 Available Models")
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if rf_model:
        st.success("✅ RF")
    else:
        st.error("❌ RF")

with col2:
    if xgb_model:
        st.success("✅ XGB")
    else:
        st.error("❌ XGB")

with col3:
    if lstm_model and tensorflow_available:
        st.success("✅ LSTM")
    else:
        st.info("ℹ️ LSTM")

st.sidebar.markdown("---")

# Location input
location_type = st.sidebar.radio("📍 Choose input type:", ["City Name", "Coordinates"])

if location_type == "City Name":
    city = st.sidebar.text_input("Enter city name:", "London")
    lat, lon, region = get_coordinates(city)
else:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        lat = st.sidebar.number_input("Latitude:", value=51.5074, format="%.4f")
    with col2:
        lon = st.sidebar.number_input("Longitude:", value=-0.1278, format="%.4f")
    region = "Custom Location"

# ============================================================================
# 6. MAIN APP
# ============================================================================

st.title("🌡️ Real-Time Weather Prediction Dashboard")
st.markdown("Powered by Machine Learning | Using Random Forest, XGBoost & LSTM")

if lat is not None and lon is not None:
    # Get current weather
    weather_data = get_weather_data(lat, lon)
    
    if weather_data:
        # Display current conditions
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🌡️ Temperature",
                f"{weather_data['temperature']:.1f}°C"
            )
        
        with col2:
            st.metric(
                "💧 Humidity",
                f"{weather_data['humidity']:.0f}%"
            )
        
        with col3:
            st.metric(
                "💨 Wind Speed",
                f"{weather_data['wind_speed']:.1f} km/h"
            )
        
        with col4:
            st.metric(
                "🔽 Pressure",
                f"{weather_data['pressure']:.1f} hPa"
            )
        
        st.markdown("---")
        
        # Prepare features
        features = prepare_features(
            weather_data['temperature'],
            weather_data['humidity'],
            weather_data['wind_speed'],
            weather_data['pressure']
        )
        
        # Predictions section
        st.subheader("🤖 Model Predictions")
        
        if st.button("🚀 Generate Forecast", use_container_width=True):
            predictions = make_predictions(features)
            
            # Display predictions
            pred_cols = st.columns(4)
            
            with pred_cols[0]:
                if predictions['RF'] is not None:
                    st.metric(
                        "🌲 Random Forest",
                        f"{predictions['RF']:.2f}°C"
                    )
                else:
                    st.metric("🌲 Random Forest", "N/A")
            
            with pred_cols[1]:
                if predictions['XGB'] is not None:
                    st.metric(
                        "⚡ XGBoost",
                        f"{predictions['XGB']:.2f}°C"
                    )
                else:
                    st.metric("⚡ XGBoost", "N/A")
            
            with pred_cols[2]:
                if predictions['LSTM'] is not None:
                    st.metric(
                        "🧠 LSTM",
                        f"{predictions['LSTM']:.2f}°C"
                    )
                else:
                    st.metric("🧠 LSTM", "Not Available")
            
            with pred_cols[3]:
                if predictions['Ensemble'] is not None:
                    st.metric(
                        "📊 Ensemble",
                        f"{predictions['Ensemble']:.2f}°C"
                    )
                else:
                    st.metric("📊 Ensemble", "N/A")
            
            st.markdown("---")
            
            # Prediction comparison chart
            valid_preds = {k: v for k, v in predictions.items() if v is not None}
            
            if valid_preds:
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    name='Predicted Temperature',
                    x=list(valid_preds.keys()),
                    y=list(valid_preds.values()),
                    marker=dict(
                        color=list(valid_preds.values()),
                        colorscale='RdYlBu_r',
                        showscale=False
                    ),
                    text=[f"{v:.2f}°C" for v in valid_preds.values()],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title="Model Predictions Comparison",
                    xaxis_title="Model",
                    yaxis_title="Temperature (°C)",
                    height=400,
                    showlegend=False,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                st.subheader("📈 Prediction Statistics")
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                values = list(valid_preds.values())
                with stats_col1:
                    st.metric("Mean Temperature", f"{np.mean(values):.2f}°C")
                with stats_col2:
                    st.metric("Min Temperature", f"{np.min(values):.2f}°C")
                with stats_col3:
                    st.metric("Max Temperature", f"{np.max(values):.2f}°C")
        
        st.markdown("---")
        
        # Info section
        st.subheader("ℹ️ About This App")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Features:
            - 🌍 Real-time weather data
            - 🤖 3 ML Models (RF, XGBoost, LSTM)
            - 📊 Ensemble averaging
            - 📈 Interactive charts
            - 🌐 Global location support
            """)
        
        with col2:
            st.markdown(f"""
            ### Current Session:
            - 📍 Location: {region}
            - 🧭 Coordinates: ({lat:.2f}°, {lon:.2f}°)
            - 🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - 🔄 Models: {sum([1 for m in [rf_model, xgb_model, lstm_model] if m is not None])} Active
            """)
        
        st.markdown("---")
        st.markdown("Built with ❤️ using Streamlit | ML Models: scikit-learn, XGBoost, TensorFlow")
