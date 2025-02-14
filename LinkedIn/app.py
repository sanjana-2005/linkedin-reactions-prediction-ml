from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

# Create Flask app
app = Flask(__name__)

# Define the path to your saved model and encoders
model_path = "C:/flaskenv/svr_pipeline_model.pkl"
encoder_dir = "C:/flaskenv/encoders"  # Path to encoders directory
scaler_path = "C:/flaskenv/scaler_linkedin.pkl"  # Path to your scaler

# Check if the model file exists at the given path
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file '{model_path}' was not found!")

# Load the trained model with joblib
try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    raise ValueError(f"An error occurred while loading the model: {e}")

# Load the label encoders (if available)
label_encoders = {}
for col in ['headline', 'location', 'media_type']:
    encoder_path = os.path.join(encoder_dir, f'{col}_encoder.pkl')
    if os.path.exists(encoder_path):
        label_encoders[col] = joblib.load(encoder_path)
        print(f"Encoder for {col} loaded successfully!")
    else:
        print(f"Encoder for {col} not found! Proceeding without encoding for {col}.")

# Load the scaler (if it exists)
scaler = None
if os.path.exists(scaler_path):
    scaler = joblib.load(scaler_path)
    print("Scaler loaded successfully!")

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the dashboard page
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the input JSON data
        data = request.get_json()

        # Convert relevant fields to numeric
        followers = float(data.get('followers', 0))
        connections = float(data.get('connections', 0))
        num_hashtags = float(data.get('num_hashtags', 0))
        comments = float(data.get('comments', 0))

        # Perform feature engineering (same as in the training step)
        followers_connections_interaction = followers * connections
        comments_hashtags_interaction = comments * num_hashtags
        followers_sq = followers ** 2
        log_comments = np.log1p(comments)
        log_followers = np.log1p(followers)

        # Construct the DataFrame based on the input structure the model expects
        df = pd.DataFrame([{
            'headline': data.get('headline', ''),
            'location': data.get('location', ''),
            'log_followers': log_followers,
            'connections': connections,
            'media_type': data.get('media_type', ''),
            'num_hashtags': num_hashtags,
            'comments': comments,
            'followers_connections_interaction': followers_connections_interaction,
            'comments_hashtags_interaction': comments_hashtags_interaction,
            'followers_sq': followers_sq,
            'log_comments': log_comments,
            'interaction_advanced': followers * num_hashtags * comments
        }])

        # Encode categorical variables using the loaded encoders (only if encoder exists)
        for col in ['headline', 'location', 'media_type']:
            le = label_encoders.get(col)
            if le:
                df[col] = le.transform(df[col].astype(str))  # Apply encoding
            else:
                print(f"Skipping encoding for {col} as encoder is missing.")

        # Check the data before scaling
        print("Data before scaling:", df.head())

        # Scale the numerical features if a scaler exists
        if scaler:
            df[['log_followers', 'connections', 'num_hashtags', 'comments',
                'followers_connections_interaction', 'comments_hashtags_interaction',
                'followers_sq', 'log_comments', 'interaction_advanced']] = scaler.transform(
                df[['log_followers', 'connections', 'num_hashtags', 'comments',
                    'followers_connections_interaction', 'comments_hashtags_interaction',
                    'followers_sq', 'log_comments', 'interaction_advanced']])

        # Check the data after scaling
        print("Data after scaling:", df.head())

        # Make prediction using the loaded model pipeline
        log_prediction = model.predict(df)

        # Inverse transform the log-transformed prediction to get the actual predicted value
        prediction = np.expm1(log_prediction[0])

        # Send the prediction back as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        # Return an error message as JSON if there is an issue
        return jsonify({'error': str(e)}), 400

# Ensure the app runs only when this file is executed directly
if __name__ == '__main__':
    app.run(debug=True)
