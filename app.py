from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

# Load pre-trained models
with open('order_complexity_model.pkl', 'rb') as f:
    complexity_model = pickle.load(f)

with open('lead_time_model.pkl', 'rb') as f:
    lead_time_model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input from form
        form_data = request.form.to_dict()
        features = pd.DataFrame([form_data], dtype=float)

        # Compute derived features
        features['Order Complexity Score'] = (
            features['Customization Level'] * 2 +
            features['Production Stages Count'] +
            features['Labor Hours Required'] * 0.5
        )
        features['Distance-Time Ratio'] = features['Distance to Delivery'] / (features['Process Time (days)'] + 1)
        features['Supplier Delay Impact'] = features['Previous Supplier Delays (%)'] / 100
        features['Priority-Customization Interaction'] = features['Order Priority'] * features['Customization Level']

        # Required features
        required_complexity_features = ['Order Size', 'Customization Level', 'Supplier Rating',
                                         'Production Stages Count', 'Labor Hours Required',
                                         'Order Complexity Score', 'Distance-Time Ratio',
                                         'Priority-Customization Interaction']
        required_lead_time_features = ['Order Size', 'Customization Level', 'Supplier Rating', 'Process Time (days)',
                                        'Shipping Mode Weight', 'Distance to Delivery', 'Stock Availability',
                                        'Production Stages Count', 'Labor Hours Required', 'Order Complexity Score',
                                        'Supplier Delay Impact', 'Priority-Customization Interaction']

        # Predict order complexity
        complexity_pred = complexity_model.predict(features[required_complexity_features])
        complexity_pred_label = "Complex" if complexity_pred[0] == 1 else "Simple"

        # Predict lead time
        lead_time_pred = lead_time_model.predict(features[required_lead_time_features])[0]

        # Prepare data for correlation calculation
        pred_data = pd.DataFrame({
            'Predicted Complexity': complexity_pred,
            'Predicted Lead Time': [lead_time_pred]
        })

        # Compute correlation
        correlation = (
            pred_data.corr().iloc[0, 1]
            if len(pred_data) > 1
            else "Not enough data for correlation"
        )

        # Feature importance
        complexity_importance = complexity_model.feature_importances_
        lead_time_importance = lead_time_model.feature_importances_

        # Store results
        results = {
            "order_complexity": complexity_pred_label,
            "predicted_lead_time": round(lead_time_pred, 2),
            "correlation": round(correlation, 2) if isinstance(correlation, float) else correlation,
            "complexity_importance": complexity_importance.tolist(),
            "lead_time_importance": lead_time_importance.tolist(),
            "complexity_features": required_complexity_features,
            "lead_time_features": required_lead_time_features,
        }

        # Render results
        return render_template('results.html', results=results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
