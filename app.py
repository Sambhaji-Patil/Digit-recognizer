from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import base64
import re
import requests

app = Flask(__name__)

# Classification reports for each model
classification_reports = {
    "cnn": """
               precision    recall  f1-score   support
           0       0.99      1.00      0.99       980
           1       1.00      1.00      1.00      1135
           2       0.99      0.99      0.99      1032
           3       0.99      1.00      0.99      1010
           4       1.00      0.99      0.99       982
           5       0.98      0.99      0.99       892
           6       1.00      0.98      0.99       958
           7       0.99      0.99      0.99      1028
           8       1.00      0.99      0.99       974
           9       0.99      0.99      0.99      1009
    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000
    """,
    "svm": """
               precision    recall  f1-score   support
           0     0.9874    0.9896    0.9885      1343
           1     0.9882    0.9925    0.9903      1600
           2     0.9706    0.9819    0.9762      1380
           3     0.9783    0.9749    0.9766      1433
           4     0.9777    0.9822    0.9800      1295
           5     0.9827    0.9796    0.9811      1273
           6     0.9858    0.9921    0.9889      1396
           7     0.9768    0.9807    0.9788      1503
           8     0.9813    0.9683    0.9748      1357
           9     0.9807    0.9669    0.9738      1420
    accuracy                         0.9810     14000
   macro avg     0.9809    0.9809    0.9809     14000
weighted avg     0.9810    0.9810    0.9810     14000
    """,
    "random_forest": """
               precision    recall  f1-score   support
           0     0.9844    0.9866    0.9855      1343
           1     0.9831    0.9831    0.9831      1600
           2     0.9522    0.9674    0.9597      1380
           3     0.9579    0.9532    0.9556      1433
           4     0.9617    0.9699    0.9658      1295
           5     0.9707    0.9631    0.9669      1273
           6     0.9800    0.9828    0.9814      1396
           7     0.9668    0.9681    0.9674      1503
           8     0.9599    0.9528    0.9564      1357
           9     0.9566    0.9465    0.9515      1420
    accuracy                         0.9675     14000
   macro avg     0.9673    0.9674    0.9673     14000
weighted avg     0.9675    0.9675    0.9675     14000
    """,
    "logistic": """
               precision    recall  f1-score   support
           0     0.9636    0.9650    0.9643      1343
           1     0.9433    0.9675    0.9553      1600
           2     0.9113    0.8935    0.9023      1380
           3     0.9021    0.8939    0.8980      1433
           4     0.9225    0.9290    0.9257      1295
           5     0.8846    0.8790    0.8818      1273
           6     0.9420    0.9534    0.9477      1396
           7     0.9273    0.9421    0.9347      1503
           8     0.8973    0.8696    0.8832      1357
           9     0.9019    0.9000    0.9010      1420
    accuracy                         0.9204     14000
   macro avg     0.9196    0.9193    0.9194     14000
weighted avg     0.9201    0.9204    0.9202     14000
    """
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_classification_report', methods=['POST'])
def get_classification_report():
    model_type = request.json['model_type']
    if model_type in classification_reports:
        return jsonify({
            'report': classification_reports[model_type]
        })
    return jsonify({'error': 'Model not found'})

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json['image']
        model_type = request.json['model_type']
        
        # Remove base64 prefix and decode the image
        img_data = re.sub('^data:image/png;base64,', '', data)
        img = Image.open(io.BytesIO(base64.b64decode(img_data))).convert("L")
        
        # Convert image to base64
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        # Make the API request
        response = requests.post(
            "https://quantumbit-mnist-classifier-api.hf.space/predict",
            json={
                "image": f"data:image/png;base64,{img_base64}",
                "model_type": model_type
            }
        )

        if response.status_code == 200:
            response_json = response.json()
            predicted_digit = response_json.get('digit')
            confidence_scores = response_json.get('confidence_scores', [])
            score_type = response_json.get('score_type', "Unknown")

            return jsonify({
                "digit": int(predicted_digit),
                "confidence_scores": confidence_scores,
                "score_type": score_type
            })
        else:
            return jsonify({"error": "Failed to get prediction from API"}), 500

def create_simulated_scores(predicted_digit):
    """Create simulated confidence scores that sum to 1.0 with highest probability for the predicted digit."""
    # Assign base probabilities
    scores = [0.01] * 10  # Give each digit a small base probability
    
    # Calculate remaining probability (should be around 0.9)
    remaining = 1.0 - sum(scores)
    
    # Assign the remaining probability to the predicted digit
    scores[predicted_digit] += remaining
    
    return scores

if __name__ == "__main__":
    app.run(debug=True)