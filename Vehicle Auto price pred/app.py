import os
from flask import Flask, render_template_string, request
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

app = Flask(__name__)

# --- Dummy ML Model Setup for Demonstration ---
# In production, replace this section by loading your saved model:
# import joblib; model = joblib.load('decision_tree_model.pkl')
X_dummy = pd.DataFrame({
    'horsepower': [110, 154, 102, 140, 160],
    'curb_weight': [2548, 2823, 2337, 2807, 3077],
    'engine_size': [130, 152, 109, 136, 151],
    'city_mpg': [21, 19, 24, 19, 17]
})
y_dummy = [13495, 16500, 13950, 17450, 17710]
model = DecisionTreeRegressor(max_depth=5)
model.fit(X_dummy, y_dummy)


# --- HTML Template with Embedded CSS ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Automobile Price Predictor</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-color: #38bdf8;
            --text-color: #f8fafc;
            --text-muted: #94a3b8;
            --border-color: #334155;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }

        .container {
            background-color: var(--card-bg);
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
            width: 100%;
            max-width: 480px;
            border: 1px solid var(--border-color);
        }

        h2 {
            margin-top: 0;
            color: var(--accent-color);
            text-align: center;
            font-size: 1.8rem;
        }

        p.subtitle {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-bottom: 30px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            font-size: 0.9rem;
        }

        input {
            width: 100%;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            background-color: #0f172a;
            color: #fff;
            font-size: 1rem;
            box-sizing: border-box;
            transition: border-color 0.3s ease;
        }

        input:focus {
            outline: none;
            border-color: var(--accent-color);
        }

        button {
            width: 100%;
            padding: 14px;
            background-color: var(--accent-color);
            color: #0f172a;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.3s ease, transform 0.1s ease;
            margin-top: 10px;
        }

        button:hover {
            background-color: #0284c7;
            color: #ffffff;
        }

        button:active {
            transform: scale(0.99);
        }

        .result-box {
            margin-top: 30px;
            padding: 20px;
            background-color: rgba(56, 189, 248, 0.1);
            border: 1px solid var(--accent-color);
            border-radius: 6px;
            text-align: center;
        }

        .result-box h3 {
            margin: 0 0 5px 0;
            font-size: 1rem;
            color: var(--text-muted);
        }

        .result-box .price {
            font-size: 2rem;
            font-weight: bold;
            color: var(--accent-color);
            margin: 0;
        }
    </style>
</head>
<body>

<div class="container">
    <h2>🚗 Auto Price Predictor</h2>
    <p class="subtitle">Decision Tree Regression Inference</p>

    <form method="POST" action="/predict">
        <div class="form-group">
            <label for="horsepower">Horsepower</label>
            <input type="number" id="horsepower" name="horsepower" step="any" placeholder="e.g. 110" required>
        </div>

        <div class="form-group">
            <label for="curb_weight">Curb Weight (lbs)</label>
            <input type="number" id="curb_weight" name="curb_weight" step="any" placeholder="e.g. 2548" required>
        </div>

        <div class="form-group">
            <label for="engine_size">Engine Size (cc)</label>
            <input type="number" id="engine_size" name="engine_size" step="any" placeholder="e.g. 130" required>
        </div>

        <div class="form-group">
            <label for="city_mpg">City MPG</label>
            <input type="number" id="city_mpg" name="city_mpg" step="any" placeholder="e.g. 21" required>
        </div>

        <button type="submit">Predict Price</button>
    </form>

    {% if prediction_text %}
    <div class="result-box">
        <h3>Estimated Vehicle Price</h3>
        <p class="price">{{ prediction_text }}</p>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from form input
        horsepower = float(request.form['horsepower'])
        curb_weight = float(request.form['curb_weight'])
        engine_size = float(request.form['engine_size'])
        city_mpg = float(request.form['city_mpg'])
        
        # Prepare feature vector
        features = np.array([[horsepower, curb_weight, engine_size, city_mpg]])
        
        # Perform prediction
        prediction = model.predict(features)[0]
        formatted_price = f"${prediction:,.2f}"
        
        return render_template_string(HTML_TEMPLATE, prediction_text=formatted_price)
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)