"""
CO2 Emission by Vehicles — Flask Web App
AIC354 Machine Learning Fundamentals | Assignment 3
"""

import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Load model bundle ─────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.pkl')

model_bundle = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model_bundle = pickle.load(f)
    print(f"✅ Model loaded: {model_bundle.get('model_name', 'Unknown')}")
else:
    print(f"⚠️  Model not found at {MODEL_PATH}. Run the notebook first.")

# ── Feature options ───────────────────────────────────────────────────────────
VEHICLE_CLASSES = [
    'COMPACT', 'FULL-SIZE', 'MID-SIZE', 'MINICOMPACT',
    'MINIVAN', 'PICKUP TRUCK - SMALL', 'PICKUP TRUCK - STANDARD',
    'SPECIAL PURPOSE VEHICLE', 'STATION WAGON - MID-SIZE',
    'STATION WAGON - SMALL', 'SUV - SMALL', 'SUV - STANDARD',
    'SUBCOMPACT', 'TWO-SEATER', 'VAN - CARGO', 'VAN - PASSENGER'
]

TRANSMISSIONS = [
    'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10',
    'AM5', 'AM6', 'AM7', 'AM8', 'AM9',
    'AS4', 'AS5', 'AS6', 'AS7', 'AS8', 'AS9', 'AS10',
    'AV', 'AV1', 'AV6', 'AV7', 'AV8', 'AV10',
    'M5', 'M6', 'M7'
]

FUEL_TYPES = {
    'X': 'Regular Gasoline',
    'Z': 'Premium Gasoline',
    'D': 'Diesel',
    'E': 'Ethanol (E85)',
    'N': 'Natural Gas',
}


@app.route('/')
def index():
    return render_template('index.html',
                           vehicle_classes=VEHICLE_CLASSES,
                           transmissions=TRANSMISSIONS,
                           fuel_types=FUEL_TYPES,
                           model_name=model_bundle.get('model_name', 'N/A') if model_bundle else 'Model not loaded')


@app.route('/predict', methods=['POST'])
def predict():
    if model_bundle is None:
        return jsonify({'error': 'Model not loaded. Please train and save the model first.'}), 500

    try:
        data = request.get_json()

        # ── Encode inputs ─────────────────────────────────────────────────────
        le_dict   = model_bundle['le_dict']
        features  = model_bundle['features']
        model     = model_bundle['model']

        vehicle_class  = data.get('vehicle_class', 'COMPACT')
        transmission   = data.get('transmission', 'A6')
        fuel_type      = data.get('fuel_type', 'X')
        engine_size    = float(data.get('engine_size', 2.0))
        cylinders      = int(data.get('cylinders', 4))
        fuel_city      = float(data.get('fuel_city', 10.0))
        fuel_hwy       = float(data.get('fuel_hwy', 8.0))
        fuel_comb      = float(data.get('fuel_comb', 9.0))
        fuel_comb_mpg  = float(data.get('fuel_comb_mpg', 30))

        # Encode categoricals using saved LabelEncoders
        def safe_encode(le, value):
            """Encode a value, defaulting to 0 if unseen."""
            classes = list(le.classes_)
            if value in classes:
                return le.transform([value])[0]
            return 0

        vc_enc   = safe_encode(le_dict['Vehicle Class'], vehicle_class)
        tr_enc   = safe_encode(le_dict['Transmission'],  transmission)
        ft_enc   = safe_encode(le_dict['Fuel Type'],     fuel_type)

        input_vector = np.array([[
            vc_enc, engine_size, cylinders, tr_enc, ft_enc,
            fuel_city, fuel_hwy, fuel_comb, fuel_comb_mpg
        ]])

        # ── Predict ───────────────────────────────────────────────────────────
        prediction = model.predict(input_vector)[0]
        prediction = max(0, round(float(prediction), 2))  # clip negative

        # Emission category
        if prediction < 120:
            category = ('🟢 Very Low', '#2ecc71')
        elif prediction < 170:
            category = ('🟡 Low', '#f1c40f')
        elif prediction < 230:
            category = ('🟠 Moderate', '#e67e22')
        elif prediction < 280:
            category = ('🔴 High', '#e74c3c')
        else:
            category = ('⛔ Very High', '#c0392b')

        return jsonify({
            'prediction'  : prediction,
            'category'    : category[0],
            'color'       : category[1],
            'unit'        : 'g/km',
            'model_used'  : model_bundle.get('model_name', 'Unknown'),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
