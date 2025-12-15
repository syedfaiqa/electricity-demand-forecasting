import pandas as pd
import numpy as np
import pickle
import os
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import bcrypt
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Replace with a secure random key in production

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_FILE = os.path.join(MODEL_DIR, "rf_model.pkl")
FALLBACK_MODEL_FILE = os.path.join(BASE_DIR, "rf_model.pkl")
DATA_FILE = os.path.join(BASE_DIR, "prediction13.csv")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)
logger.info(f"Model directory ensured at: {MODEL_DIR}")

# Globals
model = None
feature_names = []
latest_data = pd.Series()
numeric_cols = []
categorical_cols = []
_initialized = False

# Mock user database
users = {
    'admin': {
        'password': bcrypt.hashpw('password123'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    }
}

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(username):
    if username in users:
        return User(username)
    return None

# Helper: Check write permissions
def check_write_permissions(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    if not os.access(directory, os.W_OK):
        logger.warning(f"No write permission in directory: {directory}")
        return False
    logger.info(f"Write permissions verified for {directory}")
    return True

# Helper: Train and save model
def train_and_save_model(): 
    global feature_names, numeric_cols, categorical_cols
    try:
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

        logger.info(f"Loading dataset from {DATA_FILE}")
        df = pd.read_csv(DATA_FILE)
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Dataset columns: {df.columns.tolist()}")

        if "RF_Prediction" not in df.columns:
            raise ValueError("Target column 'RF_Prediction' not found in dataset.")

        df.drop(columns=["datetime", "LSTM_Prediction", "ARIMA_Prediction"], inplace=True, errors='ignore')
        df = df.ffill().bfill().fillna(0)

        X = df.drop(columns=["RF_Prediction"])
        y = df["RF_Prediction"]

        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        logger.info(f"Categorical columns: {categorical_cols}")
        logger.info(f"Numeric columns: {numeric_cols}")

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
                ('num', 'passthrough', numeric_cols)
            ]
        )

        rf_model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42))
        ])

        logger.info("Splitting data for training")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info("Training RandomForest model")
        rf_model.fit(X_train, y_train)
        logger.info("Model training completed")

        if categorical_cols:
            cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
            feature_names = cat_feature_names + numeric_cols
        else:
            feature_names = numeric_cols
        logger.info(f"Features used for training: {feature_names}")

        save_path = MODEL_FILE if check_write_permissions(MODEL_FILE) else FALLBACK_MODEL_FILE
        logger.info(f"Saving model to {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(rf_model, f)

        if os.path.exists(save_path):
            logger.info(f"Model saved successfully to {save_path}. Size: {os.path.getsize(save_path)} bytes")
        else:
            raise RuntimeError(f"Failed to save model file: {save_path}")

        return rf_model
    except Exception as e:
        logger.error(f"Error training and saving model: {str(e)}")
        raise

# Helper: Load model
def load_model():
    try:
        logger.info("Forcing model training to ensure rf_model.pkl is created")
        return train_and_save_model()
    except Exception as e:
        logger.error(f"Error loading/training model: {str(e)}")
        raise

# Initialize before first request
def initialize():
    global model, latest_data, _initialized
    if not _initialized:
        try:
            model = load_model()
            logger.info(f"Loading dataset for latest data from {DATA_FILE}")
            df = pd.read_csv(DATA_FILE)
            df.drop(columns=["datetime", "LSTM_Prediction", "ARIMA_Prediction", "RF_Prediction"], inplace=True, errors='ignore')
            df = df.ffill().bfill().fillna(0)
            if len(df) > 0:
                latest_data = df.iloc[-1].copy()
            else:
                latest_data = pd.Series({f: 0 for f in categorical_cols + numeric_cols})
            _initialized = True
            logger.info("Initialization complete.")
        except Exception as e:
            logger.error(f"Initialization error: {str(e)}")
            latest_data = pd.Series({f: 0 for f in categorical_cols + numeric_cols})

@app.before_request
def run_initialization():
    initialize()

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and bcrypt.checkpw(password.encode('utf-8'), users[username]['password'].encode('utf-8')):
            user = User(username)
            login_user(user)
            logger.info(f"User {username} logged in successfully")
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'error')
            logger.warning(f"Failed login attempt for username: {username}")
    return render_template('login.html')

# Logout route
@app.route('/logout')
@login_required
def logout():
    logout_user()
    logger.info("User logged out")
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Debug endpoint
@app.route('/debug/train', methods=['GET'])
@login_required
def debug_train():
    try:
        global model
        model = train_and_save_model()
        return jsonify({'success': True, 'message': f'Model trained and saved to {MODEL_FILE} or {FALLBACK_MODEL_FILE}'})
    except Exception as e:
        logger.error(f"Debug training error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Routes
@app.route('/')
@login_required
def home():
    try:
        return render_template('index.html', features={k: latest_data.get(k, 0.0) for k in categorical_cols + numeric_cols}, username=current_user.id)
    except Exception as e:
        logger.error(f"Home route error: {str(e)}")
        return render_template('error.html', error=str(e)), 500

@app.route('/forecasts')
@login_required
def forecasts():
    try:
        return render_template('forecasts.html', features={k: latest_data.get(k, 0.0) for k in categorical_cols + numeric_cols}, username=current_user.id)
    except Exception as e:
        logger.error(f"Forecasts route error: {str(e)}")
        return render_template('error.html', error=str(e)), 500

@app.route('/analytics')
@login_required
def analytics():
    try:
        return render_template('analytics.html', username=current_user.id)
    except Exception as e:
        logger.error(f"Analytics route error: {str(e)}")
        return render_template('error.html', error=str(e)), 500

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        user_input = {}
        for f in categorical_cols + numeric_cols:
            val = request.form.get(f, latest_data.get(f, 0))
            user_input[f] = float(val) if f in numeric_cols else str(val)
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        logger.info(f"Prediction made: {prediction:.2f} kW")
        return jsonify({
            'success': True,
            'prediction': round(prediction / 1000, 2),  # Convert kW to GW
            'message': f"Predicted Power Demand: {prediction / 1000:.2f} GW"
        })
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/forecast', methods=['POST'])
@login_required
def forecast():
    try:
        user_input = {}
        for f in categorical_cols + numeric_cols:
            val = request.form.get(f, latest_data.get(f, 0))
            user_input[f] = float(val) if f in numeric_cols else str(val)
        date_range = request.form.get('dateRange')
        start_date, end_date = date_range.split(' to ')
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        num_days = (end_date - start_date).days + 1

        predictions = []
        dates = []
        current_date = start_date
        for i in range(num_days):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0] / 1000  # Convert kW to GW
            predictions.append(round(prediction, 2))
            dates.append(current_date.strftime('%b %d'))
            current_date += timedelta(days=1)

        upper_bound = [p + 0.4 for p in predictions]
        lower_bound = [p - 0.4 for p in predictions]

        logger.info(f"Forecast generated for {num_days} days")
        return jsonify({
            'success': True,
            'dates': dates,
            'predictions': predictions,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'message': f"Forecast generated for {num_days} days"
        })
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({'success': False, 'error': 'No JSON data received'}), 400
        user_input = {}
        for f in categorical_cols + numeric_cols:
            val = data.get(f, latest_data.get(f, 0))
            user_input[f] = float(val) if f in numeric_cols else str(val)
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        logger.info(f"API prediction made: {prediction:.2f} kW")
        return jsonify({'success': True, 'prediction': round(prediction / 1000, 2), 'unit': 'GW'})
    except Exception as e:
        logger.error(f"API prediction error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/error.html')
def error_page():
    error = request.args.get('error', 'Unknown error')
    return render_template('error.html', error=error)

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)