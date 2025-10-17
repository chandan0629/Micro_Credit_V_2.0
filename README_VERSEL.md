Vercel deployment for prediction-only API

This repository contains a lightweight serverless function under `api/assess.py` that can be
deployed to Vercel if you export a small scikit-learn model (`rf_model.joblib`) to `api/`.

Steps to prepare and deploy:

1) Train and export the model locally (or on a machine with Python/scikit-learn available):

```bash
python3 -m pip install -r requirements.vercel.txt
python tools/train_and_export.py --csv credit_risk_data.csv
```

This will create `api/rf_model.joblib`.

2) Deploy to Vercel:

```bash
npm i -g vercel
vercel login
vercel --prod
```

Vercel will use `requirements.vercel.txt` to install packages for the function. Ensure the
model file `api/rf_model.joblib` is present (but do NOT commit large models to git if you prefer).

3) Test the endpoint:

```bash
curl -X POST https://your-vercel-deployment.vercel.app/api/assess \
  -H 'Content-Type: application/json' \
  -d '{"total_transactions":10000,"avg_transaction_amount":1500,"payment_consistency_score":90,"business_age_months":48,"digital_footprint_score":85}'
```

Notes:
- Vercel serverless functions have size/time limits. Keep the model small (<50-100MB) and dependencies minimal.
- For full training or heavy TF inference keep using the Docker/VPS approach.
