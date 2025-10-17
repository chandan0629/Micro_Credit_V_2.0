import json
import os
from joblib import load

# Load the small scikit-learn model at cold start
MODEL_FILENAME = 'rf_model.joblib'
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

try:
    model = load(MODEL_PATH)
except Exception:
    model = None

def handler(request):
    """Vercel serverless function entrypoint.

    Expects JSON body with the five feature fields and returns a small JSON response.
    """
    if model is None:
        return {
            'statusCode': 500,
            'body': json.dumps({'success': False, 'error': 'Model not found. Run the export script to generate the model.'})
        }

    try:
        data = request.json()

        required = ['total_transactions', 'avg_transaction_amount', 'payment_consistency_score',
                    'business_age_months', 'digital_footprint_score']
        for f in required:
            if f not in data:
                return {'statusCode': 400, 'body': json.dumps({'success': False, 'error': f'Missing field: {f}'})}

        features = [
            data['total_transactions'],
            data['avg_transaction_amount'],
            data['payment_consistency_score'],
            data['business_age_months'],
            data['digital_footprint_score']
        ]

        proba = float(model.predict_proba([features])[0][1])

        resp = {
            'success': True,
            'risk_score': proba,
            'risk_percentage': round(proba * 100, 2)
        }

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps(resp)
        }
    except Exception as e:
        return {'statusCode': 500, 'body': json.dumps({'success': False, 'error': str(e)})}
