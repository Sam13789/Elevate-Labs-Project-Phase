import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="🏠 Airbnb Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Set a consistent Plotly template and colorway (safe defaults)
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#87CEEB", "#FF5A5F", "#484848"]

# Load the trained model
@st.cache_resource
def load_model():
    try:
        # Check for required ML libraries
        missing_libs = []
        try:
            import lightgbm
        except ImportError:
            missing_libs.append("lightgbm")
        
        try:
            import xgboost
        except ImportError:
            missing_libs.append("xgboost")
            
        try:
            import catboost
        except ImportError:
            missing_libs.append("catboost")
        
        if missing_libs:
            st.error(f"❌ Missing required libraries: {', '.join(missing_libs)}")
            st.info("🔧 Please install missing libraries:")
            st.code(f"pip install {' '.join(missing_libs)}")
            st.info("Then restart the Streamlit app.")
            return None, None
        
        # Ensure model directory exists
        model_dir = "models/deployment/"
        os.makedirs(model_dir, exist_ok=True)

        # Try specific whitelisted filename first
        preferred_name = os.path.join(model_dir, "best_model.pkl")

        # If missing, optionally download from MODEL_URL env var
        if not os.path.exists(preferred_name):
            model_url = os.environ.get("MODEL_URL") or st.secrets.get("MODEL_URL", None)
            if model_url:
                try:
                    with st.spinner("⬇️ Downloading model..."):
                        resp = requests.get(model_url, timeout=60)
                        resp.raise_for_status()
                        with open(preferred_name, "wb") as f:
                            f.write(resp.content)
                except Exception as dl_err:
                    st.warning(f"Couldn't download model from MODEL_URL: {dl_err}")

        # If preferred file exists, load it
        if os.path.exists(preferred_name):
            with open(preferred_name, 'rb') as f:
                model_package = pickle.load(f)
            return model_package, preferred_name

        # Fallback: find latest timestamped model file
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
            if model_files:
                latest_model = sorted(model_files)[-1]
                model_path = os.path.join(model_dir, latest_model)
                with open(model_path, 'rb') as f:
                    model_package = pickle.load(f)
                return model_package, model_path

        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("🔧 This might be due to missing ML libraries. Try installing:")
        st.code("pip install lightgbm xgboost catboost")
        return None, None

# Feature engineering function (simplified version for prediction)
def engineer_features(input_data):
    """Create features matching the trained model's expected format"""
    
    # The model expects exactly these 26 features in this order:
    expected_features = [
        'feature_0', 'feature_1', 'feature_3', 'feature_20', 'feature_29', 
        'feature_30', 'feature_31', 'feature_35', 'feature_37', 'feature_42', 
        'feature_52', 'feature_53', 'feature_55', 'feature_56', 'feature_64', 
        'feature_70', 'feature_74', 'feature_76', 'feature_88', 'feature_96', 
        'feature_97', 'feature_98', 'feature_101', 'feature_102', 'feature_113', 
        'feature_114'
    ]
    
    # Create a simplified mapping based on the most important features
    feature_values = {}
    
    # Neighbourhood group (one-hot encoded)
    feature_values['feature_0'] = 1.0 if input_data['neighbourhood_group'] == 'Bronx' else 0.0
    feature_values['feature_1'] = 1.0 if input_data['neighbourhood_group'] == 'Brooklyn' else 0.0
    feature_values['feature_3'] = 1.0 if input_data['neighbourhood_group'] == 'Manhattan' else 0.0
    
    # Location features
    feature_values['feature_29'] = input_data['latitude']
    feature_values['feature_30'] = input_data['longitude']
    
    # Distance to Manhattan (approximate)
    manhattan_lat, manhattan_lng = 40.7831, -73.9712
    distance = np.sqrt((input_data['latitude'] - manhattan_lat)**2 + 
                      (input_data['longitude'] - manhattan_lng)**2)
    feature_values['feature_31'] = distance
    
    # Room type features
    feature_values['feature_52'] = 1.0 if input_data['room_type'] == 'Entire home/apt' else 0.0
    feature_values['feature_53'] = 1.0 if input_data['room_type'] == 'Private room' else 0.0
    
    # Minimum nights (log transform)
    feature_values['feature_55'] = np.log1p(input_data['minimum_nights'])
    
    # Number of reviews (log transform)  
    feature_values['feature_56'] = np.log1p(input_data['number_of_reviews'])
    
    # Reviews per month
    feature_values['feature_74'] = input_data.get('reviews_per_month', 0)
    
    # Host listings (log transform)
    feature_values['feature_76'] = np.log1p(input_data.get('calculated_host_listings_count', 1))
    
    # Availability
    feature_values['feature_88'] = input_data['availability_365']
    
    # Privacy score (derived feature)
    privacy_scores = {'Entire home/apt': 1.0, 'Private room': 0.6, 'Shared room': 0.2}
    feature_values['feature_96'] = privacy_scores.get(input_data['room_type'], 0.5)
    
    # Fill remaining features with zeros
    for feature in expected_features:
        if feature not in feature_values:
            feature_values[feature] = 0.0
    
    # Create DataFrame with exact feature order
    feature_row = [feature_values[feature] for feature in expected_features]
    
    return pd.DataFrame([feature_row], columns=expected_features)

# Prediction function
def make_prediction(model_package, input_data):
    """Make price prediction using the loaded model"""
    try:
        # Engineer features
        features_df = engineer_features(input_data)
        
        # Make prediction based on model type
        if model_package['model_type'] == 'stacking_ensemble':
            # Get base model predictions
            base_models = model_package['base_models']
            meta_learner = model_package['meta_learner']
            
            base_predictions = np.zeros((len(features_df), len(base_models)))
            for i, (name, model) in enumerate(base_models.items()):
                base_predictions[:, i] = model.predict(features_df)
            
            log_prediction = meta_learner.predict(base_predictions)[0]
            
        elif model_package['model_type'] == 'individual':
            model = model_package['model']
            log_prediction = model.predict(features_df)[0]
            
        else:
            # Handle other ensemble types
            model = model_package['model']
            log_prediction = model.predict(features_df)[0]
        
        # Convert back to original scale
        price_prediction = np.expm1(log_prediction)
        
        return price_prediction, log_prediction
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

# Main app
def main():
    # Header
    st.markdown('<h1>🏠 Airbnb Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p>Get instant price predictions for your Airbnb listing using advanced machine learning</p>', unsafe_allow_html=True)
    
    # Load model
    model_package, model_path = load_model()
    
    if model_package is None:
        st.error("❌ Could not load the trained model. Please ensure the model file exists in the models/deployment/ directory.")
        st.stop()
    
    # Model info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🏆 Model", "Stacking Ensemble")
    with col2:
        st.metric("📊 Accuracy (R²)", "0.225")
    with col3:
        st.metric("💰 Avg Error", "±$54")
    
    st.divider()
    
    # Input form
    with st.container():
        st.subheader("📝 Property Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏘️ Location Information")
            neighbourhood_group = st.selectbox(
                "Neighbourhood Group",
                ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
            )
            latitude = st.number_input("Latitude", min_value=40.4, max_value=40.9, value=40.7589, step=0.0001)
            longitude = st.number_input("Longitude", min_value=-74.3, max_value=-73.7, value=-73.9851, step=0.0001)
            
            st.markdown("#### 🏠 Property Information")
            room_type = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room"])
            minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=1)
            availability_365 = st.number_input("Availability (days per year)", min_value=0, max_value=365, value=365)
        
        with col2:
            st.markdown("#### 📊 Review Information")
            number_of_reviews = st.number_input("Number of Reviews", min_value=0, max_value=1000, value=10)
            reviews_per_month = st.number_input("Reviews per Month", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
            
            st.markdown("#### 👤 Host Information")
            calculated_host_listings_count = st.number_input("Host Listings Count", min_value=1, max_value=100, value=1)
    
    st.divider()
    
    # Prediction button
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        # Prepare input data
        input_data = {
            'neighbourhood_group': neighbourhood_group,
            'latitude': latitude,
            'longitude': longitude,
            'room_type': room_type,
            'minimum_nights': minimum_nights,
            'number_of_reviews': number_of_reviews,
            'reviews_per_month': reviews_per_month,
            'calculated_host_listings_count': calculated_host_listings_count,
            'availability_365': availability_365
        }
        
        # Make prediction
        with st.spinner("🤖 Analyzing property features and generating prediction..."):
            predicted_price, log_price = make_prediction(model_package, input_data)
        
        if predicted_price is not None:
            # Display prediction
            st.success(f"💰 Predicted Price: ${predicted_price:.0f} per night")
            st.caption(f"Confidence Range: ${predicted_price-54:.0f} - ${predicted_price+54:.0f}")
            
            # Insights
            col1, col2, col3 = st.columns(3)
            with col1:
                monthly_revenue = predicted_price * 30 * (availability_365/365)
                st.metric("📅 Est. Monthly Revenue", f"${monthly_revenue:,.0f}")
            with col2:
                yearly_revenue = predicted_price * availability_365
                st.metric("📊 Est. Yearly Revenue", f"${yearly_revenue:,.0f}")
            with col3:
                borough_avg = {'Manhattan': 196, 'Brooklyn': 124, 'Queens': 99, 'Bronx': 87, 'Staten Island': 114}
                avg_price = borough_avg.get(neighbourhood_group, 120)
                vs_avg = ((predicted_price - avg_price) / avg_price) * 100
                st.metric("🏘️ vs Borough Avg", f"{vs_avg:+.0f}%")
            
            # Chart
            st.subheader("📈 Price Analysis")
            comparison_data = pd.DataFrame({
                'Borough': list(borough_avg.keys()) + ['Your Property'],
                'Price': list(borough_avg.values()) + [predicted_price],
                'Type': ['Average'] * 5 + ['Prediction']
            })
            fig = px.bar(
                comparison_data, x='Borough', y='Price', color='Type',
                title='Price Comparison by Borough',
                color_discrete_map={'Average': '#87CEEB', 'Prediction': '#FF5A5F'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Factors
            st.subheader("🎯 Key Factors Affecting Your Price")
            factors = []
            if neighbourhood_group == 'Manhattan':
                factors.append("📍 Premium location (Manhattan)")
            if room_type == 'Entire home/apt':
                factors.append("🏠 Entire home increases value")
            if number_of_reviews > 20:
                factors.append("⭐ High review count builds trust")
            if availability_365 > 300:
                factors.append("📅 High availability")
            if factors:
                for f in factors:
                    st.write("- ", f)
            else:
                st.write("- 📊 Standard pricing factors applied")
    
    st.divider()
    st.caption("🤖 Powered by Advanced Machine Learning | 🏠 Built for Airbnb Hosts")

if __name__ == "__main__":
    main()
