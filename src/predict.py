import pickle
import numpy as np
from flask import Flask, request, jsonify
from datetime import datetime

# File paths
dv_path = r"./models/dv.bin"
model_path = r"./models/model.bin"

# Load files


def load_file(file):
    with open(file, 'rb') as f_in:
        return pickle.load(f_in)


dv = load_file(dv_path)
model = load_file(model_path)

# Initialize Flask app
app = Flask("hdb-resale-flat-price")


@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Extract input data
        data = request.json
        lease_commence_year = int(data['lease_commence_year'])
        lease_commence_month = int(
            data.get('lease_commence_month', 6))  # Default to June
        floor_area_sqm = float(data['floor_area_sqm'])
        town = data['town']
        flat_type = data['flat_type']
        flat_model = data['flat_model']
        storey_range = data['storey_range']

        # Calculate remaining lease
        current_date = datetime.now()
        current_year = current_date.year
        current_month = current_date.month
        remaining_lease = 99 - ((current_year - lease_commence_year) +
                                (current_month - lease_commence_month) / 12)

        # Prepare feature dictionary
        features = {
            "town": town,
            "flat_type": flat_type,
            "flat_model": flat_model,
            "storey_range": storey_range,
            "floor_area_sqm": floor_area_sqm,
            "remaining_lease": remaining_lease
        }

        # Transform features using DictVectorizer
        X = dv.transform([features])

        # Predict
        y_pred = model.predict(X)
        predicted_price = np.expm1(y_pred[0])  # Reverse log transformation
        predicted_price = int(predicted_price.round(0))

        # Return result
        result = {
            "predicted_price": predicted_price
        }
        return jsonify(result)

    except KeyError as e:
        return jsonify({"error": f"Missing input field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
