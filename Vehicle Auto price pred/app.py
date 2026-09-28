from flask import Flask, render_template_string, request
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# --- DATA LOADING & MODEL TRAINING ---
def load_data_and_model():
    df = pd.read_csv('autos_dataset.csv')
    
    # Preprocessing
    df['num-of-cylinders'] = df['num-of-cylinders'].replace({
        'four': 4, 'six': 6, 'five': 5, 'three': 3, 
        'twelve': 12, 'two': 2, 'eight': 8
    })
    df.replace({'?': np.nan}, inplace=True)
    
    drop_cols = ['make', 'fuel-type', 'aspiration', 'engine-type', 'normalized-losses',
                 'num-of-doors', 'bore', 'stroke', 'fuel-system',
                 'body-style', 'drive-wheels', 'engine-location']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)
    
    df['price'] = df['price'].astype(float)
    df['peak-rpm'] = df['peak-rpm'].astype(float)
    df['horsepower'] = df['horsepower'].astype(float)
    
    df['price'] = df['price'].fillna(df['price'].mean())
    df['peak-rpm'] = df['peak-rpm'].fillna(df['peak-rpm'].mean())
    df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())
    
    X = df.drop(['price'], axis=1)
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeRegressor(max_depth=7, max_leaf_nodes=50, min_samples_leaf=2, splitter='random', random_state=42)
    model.fit(X_train, y_train)
    
    return df, model

df, model = load_data_and_model()

# --- HTML TEMPLATE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vehicle Auto Price Prediction</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: 'Helvetica Neue', Arial, sans-serif;
            /* Lighter background gradient overlay so the car image pops through */
            background-image: linear-gradient(rgba(15, 23, 42, 0.45), rgba(15, 23, 42, 0.65)), 
                              url('https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=2000&q=80');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #f8fafc;
        }
        .container {
            max-width: 900px;
            margin: 40px auto;
            /* Semi-transparent container with a frosted glass effect (backdrop-filter) */
            background: rgba(30, 41, 59, 0.65);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.6);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        h1 {
            text-align: center;
            color: #f8fafc;
            margin-bottom: 5px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        p.subtitle {
            text-align: center;
            color: #e2e8f0;
            margin-bottom: 30px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 20px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-size: 14px;
            color: #f1f5f9;
            font-weight: 500;
        }
        input, select {
            width: 100%;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            background-color: rgba(15, 17, 23, 0.7);
            color: white;
            box-sizing: border-box;
        }
        input:focus, select:focus {
            outline: none;
            border-color: #f59e0b;
            background-color: rgba(15, 17, 23, 0.9);
        }
        button {
            width: 100%;
            padding: 12px;
            background-color: #f59e0b;
            color: #0f172a;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 20px;
            transition: background 0.3s, transform 0.2s;
            box-shadow: 0 4px 10px rgba(245, 158, 11, 0.4);
        }
        button:hover {
            background-color: #d97706;
            transform: translateY(-1px);
        }
        .result-box {
            margin-top: 25px;
            background-color: rgba(16, 185, 129, 0.3);
            backdrop-filter: blur(5px);
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #10b981;
            text-align: center;
        }
        .result-box h2 {
            color: #6ee7b7;
            margin: 0 0 10px 0;
        }
        .result-box h1 {
            color: #ffffff;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.6);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 Vehicle Auto Price Prediction</h1>
        <p class="subtitle">Machine Learning Valuation Tool</p>
        
        <form method="POST">
            <div class="grid">
                <div class="form-group">
                    <label>Insurance Risk Rating (Symboling)</label>
                    <select name="symboling">
                        <option value="0">0 (Average)</option>
                        <option value="-3">-3 (Very Safe)</option>
                        <option value="-2">-2</option>
                        <option value="-1">-1</option>
                        <option value="1">1</option>
                        <option value="2">2</option>
                        <option value="3">3 (Risky)</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Wheel Base</label>
                    <input type="number" step="any" name="wheel_base" value="{{ defaults['wheel-base'] }}" required>
                </div>
                <div class="form-group">
                    <label>Vehicle Length</label>
                    <input type="number" step="any" name="length" value="{{ defaults['length'] }}" required>
                </div>
                <div class="form-group">
                    <label>Vehicle Width</label>
                    <input type="number" step="any" name="width" value="{{ defaults['width'] }}" required>
                </div>
                <div class="form-group">
                    <label>Vehicle Height</label>
                    <input type="number" step="any" name="height" value="{{ defaults['height'] }}" required>
                </div>
                <div class="form-group">
                    <label>Curb Weight (lbs)</label>
                    <input type="number" name="curb_weight" value="{{ defaults['curb-weight'] }}" required>
                </div>
                <div class="form-group">
                    <label>Number of Cylinders</label>
                    <select name="num_cylinders">
                        <option value="4" selected>4</option>
                        <option value="2">2</option>
                        <option value="3">3</option>
                        <option value="5">5</option>
                        <option value="6">6</option>
                        <option value="8">8</option>
                        <option value="12">12</option>
                    </select>
                </div>
                <div class="form-group">
                    <label>Engine Size (cu. in)</label>
                    <input type="number" name="engine_size" value="{{ defaults['engine-size'] }}" required>
                </div>
                <div class="form-group">
                    <label>Compression Ratio</label>
                    <input type="number" step="any" name="compression_ratio" value="{{ defaults['compression-ratio'] }}" required>
                </div>
                <div class="form-group">
                    <label>Horsepower</label>
                    <input type="number" step="any" name="horsepower" value="{{ defaults['horsepower'] }}" required>
                </div>
                <div class="form-group">
                    <label>Peak RPM</label>
                    <input type="number" step="any" name="peak_rpm" value="{{ defaults['peak-rpm'] }}" required>
                </div>
                <div class="form-group">
                    <label>City MPG</label>
                    <input type="number" name="city_mpg" value="{{ defaults['city-mpg'] }}" required>
                </div>
                <div class="form-group">
                    <label>Highway MPG</label>
                    <input type="number" name="highway_mpg" value="{{ defaults['highway-mpg'] }}" required>
                </div>
            </div>
            
            <button type="submit">Calculate Estimated Vehicle Price</button>
        </form>

        {% if prediction is not none %}
        <div class="result-box">
            <h2>Estimated Market Value</h2>
            <h1>${{ "{:,.2f}".format(prediction) }}</h1>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    defaults = df.mean(numeric_only=True).to_dict()
    
    if request.method == 'POST':
        try:
            features = [
                float(request.form['symboling']),
                float(request.form['wheel_base']),
                float(request.form['length']),
                float(request.form['width']),
                float(request.form['height']),
                float(request.form['curb_weight']),
                float(request.form['num_cylinders']),
                float(request.form['engine_size']),
                float(request.form['compression_ratio']),
                float(request.form['horsepower']),
                float(request.form['peak_rpm']),
                float(request.form['city_mpg']),
                float(request.form['highway_mpg'])
            ]
            prediction = model.predict([features])[0]
        except Exception as e:
            print(f"Error making prediction: {e}")
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction, defaults=defaults)

if __name__ == '__main__':
    app.run(debug=True)