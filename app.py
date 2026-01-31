from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO

# Set up matplotlib to avoid GUI issues
import matplotlib

matplotlib.use('Agg')

app = Flask(__name__)

# Global variables for model and preprocessor
model = None
preprocessor = None
feature_names = None
metadata = None
shap_available = False


def load_model_and_preprocessor():
    # Loading the trained model and preprocessor
    global model, preprocessor, feature_names, metadata, shap_available

    try:
        # Load metadata
        with open('prepared_data/metadata.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        feature_names = metadata['feature_names']

        # Load preprocessor
        preprocessor = joblib.load('prepared_data/preprocessor.joblib')

        # Load best model info
        with open('models/best_model_info.json', 'r', encoding='utf-8') as f:
            best_model_info = json.load(f)

        best_model_name = best_model_info['best_model_name']
        model_dir = f"models/{best_model_name.replace(' ', '_').lower()}"

        # Load the model
        if best_model_name == 'Artificial Neural Network':
            import tensorflow as tf
            model = tf.keras.models.load_model(f'{model_dir}/model.h5')
        else:
            model = joblib.load(f'{model_dir}/model.joblib')

        # Check if SHAP analysis is available
        shap_available = os.path.exists('shap_results/shap_summary_bar.png')

        print(f"Model loaded: {best_model_name}")
        print(f"Features: {len(feature_names)}")
        print(f"SHAP available: {shap_available}")

        return True

    except Exception as e:
        print(f"Error loading model: {e}")
        return False


# Load model on startup
if not load_model_and_preprocessor():
    print("Could not load model. Please run the training pipeline first.")


def predict_churn(customer_data):
    # Predicting the churn probability for a customer
    try:
        # Create DataFrame from customer data
        df = pd.DataFrame([customer_data])

        # Preprocess the data
        processed_data = preprocessor.transform(df)

        # Make prediction
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(processed_data)[0][1]
            prediction = int(probability > 0.5)
        else:
            # For neural networks
            probability = model.predict(processed_data, verbose=0)[0][0]
            prediction = int(probability > 0.5)

        return probability, prediction, processed_data

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def generate_shap_explanation(customer_data, processed_data):
    # Generating Shap explanations for the prediction
    try:
        print("Generating SHAP explanation ..")

        # Guard: need processed_data and model
        if processed_data is None or model is None:
            print("Processed data or model missing — skipping SHAP")
            return {'contributions': [], 'total_positive': 0.0, 'total_negative': 0.0, 'available': False}

        # Choose explainer based on model type
        if hasattr(model, 'predict_proba'):
            print("Using TreeExplainer for tree-based model")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(processed_data)
            # If list returned (per-class), pick positive class for binary
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    shap_values = shap_values[1]
                else:
                    shap_values = shap_values[0]
        else:
            print("Using KernelExplainer for non-tree model (may be slower)")
            explainer = shap.KernelExplainer(model.predict, processed_data)
            shap_values = explainer.shap_values(processed_data, nsamples=50)

        # Optional expected value
        expected_value = None
        if hasattr(explainer, 'expected_value'):
            try:
                ev = explainer.expected_value
                if isinstance(ev, np.ndarray):
                    if len(ev) == 2:
                        expected_value = float(ev[1])
                    elif len(ev) > 0:
                        expected_value = float(ev[0])
                else:
                    expected_value = float(ev)
            except Exception:
                expected_value = None

        # Convert to numpy and reduce to 1D for first sample
        shap_array = np.array(shap_values)

        if shap_array.ndim == 3:
            # (1, classes, features) -> pick positive class if present
            if shap_array.shape[1] == 2:
                shap_array = shap_array[0, 1, :]
            else:
                shap_array = shap_array[0]
        elif shap_array.ndim == 2:
            shap_array = shap_array[0] if shap_array.shape[0] == 1 else shap_array[0]
        elif shap_array.ndim == 1:
            pass
        else:
            shap_array = shap_array.reshape(-1)

        shap_array = np.asarray(shap_array).squeeze()
        print(f"Processed SHAP array shape: {shap_array.shape}")

        # Determine feature names
        if 'feature_names' in globals() and feature_names:
            fnames = feature_names
        else:
            if hasattr(processed_data, 'columns'):
                fnames = list(processed_data.columns)
            else:
                fnames = [f'feature_{i}' for i in range(len(shap_array))]

        contributions = []
        for i, fname in enumerate(fnames):
            if i < len(shap_array):
                value = shap_array[i]
                contribution = float(np.sum(value))

                contributions.append({
                    'feature': fname,
                    'simplified_name': simplify_feature_name(fname, metadata),
                    'contribution': contribution,
                    'abs_contribution': abs(contribution),
                    'direction': 'increases' if contribution > 0 else 'decreases'
                })

        # Sort by absolute contribution descending
        contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)

        total_positive = sum([c['contribution'] for c in contributions if c['contribution'] > 0])
        total_negative = sum([c['contribution'] for c in contributions if c['contribution'] < 0])

        print(f"Total positive contributions: {total_positive}")
        print(f"Total negative contributions: {total_negative}")
        print(f"SHAP contributions computed: {len(contributions)} features")

        return {
            'contributions': contributions[:10],  # top 10
            'total_positive': float(total_positive),
            'total_negative': float(total_negative),
            'available': True
        }

    except Exception as e:
        print(f"SHAP explanation error (numeric): {e}")
        import traceback
        traceback.print_exc()
        return {
            'contributions': [],
            'available': False,
            'error': str(e)
        }

def simplify_feature_name(feature_name, metadata):
    # Simplify feature names for display
    # Map original names to display names
    display_names = {
        'Ages': 'Age',
        'ServicesOpted': 'Services Opted',
        'FrequentFlyer_No': 'Not a Frequent Flyer (vs Yes)',
        'FrequentFlyer_No Record': 'No Frequent Flyer Record (vs Yes)',
        'FrequentFlyer_Yes': 'Frequent Flyer',
        'AnnualIncomeClass_High Income': 'High Income (vs Low Income)',
        'AnnualIncomeClass_Middle Income': 'Middle Income (vs Low Income)',
        'AccountSyncedToSocialMedia_Yes': 'Social Media Connected (vs Not Connected)',
        'BookedHotelOrNot_Yes': 'Booked Hotel (vs Not Booked)'
    }

    # Return display name if available
    if feature_name in display_names:
        return display_names[feature_name]

    # Handle categorical features
    if '_' in feature_name:
        parts = feature_name.split('_')
        if len(parts) > 1:
            base_feature = parts[0]
            value = '_'.join(parts[1:])

            # Check if we have category info
            if 'category_info' in metadata and base_feature in metadata['category_info']:
                ref_category = metadata['category_info'][base_feature]['reference_category']

                # Convert base feature to display name
                base_display = {
                    'FrequentFlyer': 'Frequent Flyer',
                    'AnnualIncomeClass': 'Income Class',
                    'AccountSyncedToSocialMedia': 'Social Media Sync',
                    'BookedHotelOrNot': 'Hotel Booking'
                }.get(base_feature, base_feature)

                return f"{base_display}: {value} (vs {ref_category})"

    return feature_name


def generate_recommendations(probability, customer_data, top_risk_factors):
    # Generating personalized recommendations based on prediction
    risk_level = "LOW"
    risk_color = "#28a745"

    if probability >= 0.8:
        risk_level = "VERY HIGH"
        risk_color = "#dc3545"
    elif probability >= 0.6:
        risk_level = "HIGH"
        risk_color = "#fd7e14"
    elif probability >= 0.4:
        risk_level = "MEDIUM"
        risk_color = "#ffc107"

    # Base recommendation
    if probability >= 0.6:
        recommendation = "Immediate action required. Contact customer within 24 hours."
    elif probability >= 0.4:
        recommendation = "Proactive outreach recommended. Schedule call within 48 hours."
    else:
        recommendation = "Monitor customer. Include in next retention campaign."

    # Generate specific recommendations based on customer data
    specific_recommendations = []

    # Extract customer data
    age = customer_data.get('Ages')
    income = customer_data.get('AnnualIncomeClass')
    services = customer_data.get('ServicesOpted')
    frequent_flyer = customer_data.get('FrequentFlyer')
    social_sync = customer_data.get('AccountSyncedToSocialMedia')
    booked_hotel = customer_data.get('BookedHotelOrNot')

    # Age-based recommendations
    if age and age < 30:
        specific_recommendations.append("Target with youth-focused marketing campaigns and mobile app promotions")
    elif age and age > 50:
        specific_recommendations.append("Offer senior discounts, personalized service, and easy-to-use interfaces")

    # Income-based recommendations
    if income == "Low Income":
        specific_recommendations.append("Provide budget-friendly service bundles and flexible payment options")
    elif income == "Middle Income":
        specific_recommendations.append("Offer family packages and value-added services")
    elif income == "High Income":
        specific_recommendations.append("Provide premium concierge services and exclusive benefits")

    # Services-based recommendations
    if services and services <= 2:
        specific_recommendations.append("Upsell complementary services with 30-day free trial")
    elif services and services >= 5:
        specific_recommendations.append("Create custom bundle discount for loyalty and service optimization review")

    # Frequent Flyer recommendations
    if frequent_flyer == "Yes":
        specific_recommendations.append("Reward with frequent flyer loyalty points and travel upgrades")
    elif frequent_flyer == "No Record":
        specific_recommendations.append("Complete travel profile for personalized offers and travel preferences survey")
    elif frequent_flyer == "No":
        specific_recommendations.append("Introduce travel benefits program with sign-up bonus")

    # Social media recommendations
    if social_sync == "Yes":
        specific_recommendations.append("Engage through social media exclusive offers and referral program")
    else:
        specific_recommendations.append("Encourage social media connection for exclusive deals")

    # Hotel booking recommendations
    if booked_hotel == "Yes":
        specific_recommendations.append("Offer hotel loyalty rewards and room upgrade opportunities")
    else:
        specific_recommendations.append("Promote hotel booking packages with special discounts")

    # Add SHAP-based recommendations from top risk factors
    for factor in top_risk_factors[:3]:
        feature = factor['simplified_name'].lower()
        if 'age' in feature and 'young' not in specific_recommendations[0].lower():
            specific_recommendations.append("Consider age-appropriate engagement strategies")
        elif 'income' in feature and 'low' in feature:
            specific_recommendations.append("Review pricing strategy for affordability")
        elif 'service' in feature:
            specific_recommendations.append("Conduct service satisfaction survey")

    return {
        'risk_level': risk_level,
        'risk_color': risk_color,
        'recommendation': recommendation,
        'specific_recommendations': specific_recommendations[:5]  # Top 5
    }


def load_model_info():
    # Load model information for display
    try:
        with open('models/best_model_info.json', 'r', encoding='utf-8') as f:
            best_model_info = json.load(f)

        # Load model metrics
        best_model_name = best_model_info['best_model_name']
        model_dir = f"models/{best_model_name.replace(' ', '_').lower()}"

        metrics_path = f'{model_dir}/metrics.json'
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)

        return {
            'best_model': best_model_name,
            'best_metrics': metrics,
            'top_features': best_model_info.get('best_model_top_features', []),
            'loaded': True
        }
    except Exception as e:
        print(f" Error loading model info: {e}")
        return {
            'loaded': False,
            'error': str(e)
        }


@app.route('/')
def index():
    # Rendering the default page
    model_info = load_model_info() if model is not None else {'loaded': False}

    return render_template('index.html',
                           model_loaded=model is not None,
                           model_info=model_info,
                           shap_available=shap_available)


@app.route('/predict', methods=['POST'])
def predict():
    # Handling prediction requests
    try:
        # Get data from frontend (with correct column names)
        customer_data = request.json

        # Validate input - using CSV column names
        required_fields = ['Ages', 'FrequentFlyer', 'AnnualIncomeClass',
                           'ServicesOpted', 'AccountSyncedToSocialMedia', 'BookedHotelOrNot']

        for field in required_fields:
            if field not in customer_data:
                return jsonify({
                    'success': False,
                    'error': f'Missing field: {field}'
                }), 400

        # Convert to proper types
        customer_data['Ages'] = int(customer_data['Ages'])
        customer_data['ServicesOpted'] = int(customer_data['ServicesOpted'])

        # Make prediction
        probability, prediction, processed_data = predict_churn(customer_data)

        if probability is None:
            return jsonify({
                'success': False,
                'error': 'Prediction failed'
            }), 500

        # Generate SHAP explanation
        shap_explanation = generate_shap_explanation(customer_data, processed_data)

        # Get top risk factors from SHAP
        top_risk_factors = shap_explanation.get('contributions', [])

        # Generate recommendations
        recommendations = generate_recommendations(probability, customer_data, top_risk_factors)

        # Load model info
        model_info_data = load_model_info()

        # Return response with user-friendly field names for display
        response = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'customer': {
                'age': customer_data['Ages'],
                'frequent_flyer': customer_data['FrequentFlyer'],
                'income_class': customer_data['AnnualIncomeClass'],
                'services_opted': customer_data['ServicesOpted'],
                'social_sync': customer_data['AccountSyncedToSocialMedia'],
                'booked_hotel': customer_data['BookedHotelOrNot']
            },
            'prediction': {
                'probability': float(probability),
                'predicted_class': int(prediction),
                'risk_level': recommendations['risk_level'],
                'risk_color': recommendations['risk_color'],
                'recommendation': recommendations['recommendation']
            },
            'recommendations': recommendations['specific_recommendations'],
            'shap': shap_explanation,
            'model_info': {
                'name': model_info_data.get('best_model', 'Unknown'),
                'accuracy': model_info_data.get('best_metrics', {}).get('accuracy', 0)
            }
        }

        return jsonify(response)

    except Exception as e:
        print(f" Error in predict endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/model/info')
def model_info():
    # Return model information
    model_info_data = load_model_info()

    return jsonify({
        'success': model_info_data['loaded'],
        'loaded': model is not None,
        'model_info': model_info_data,
        'shap_available': shap_available
    })


@app.route('/health')
def health():
    # Health check endpoint
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('templates', exist_ok=True)

    # Run the app
    print("Starting Flask application...")
    print(f"Model loaded: {model is not None}")
    print(f"SHAP available: {shap_available}")
    print("Server running at http://localhost:5000")

    app.run(debug=True, host='127.0.0.1', port=5000,use_reloader=False)