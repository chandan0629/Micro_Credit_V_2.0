# Micro_Credit_V_2.0


# 🏦 AI-Powered Micro Credit Assessment System

A comprehensive micro credit system with multi-model risk assessment and real-time loan recommendations.

## 🚀 Features

### Backend (Python Flask API)
- **5 AI Models**: Neural Network, Random Forest, Gradient Boosting, Logistic Regression, SVM
- **Model Comparison**: Real-time performance comparison across all models
- **Ensemble Predictions**: Combined predictions for improved accuracy
- **RESTful API**: Clean REST endpoints for all operations
- **Data Export**: CSV export functionality for assessments
- **Assessment History**: Track and analyze past assessments

### Frontend (Interactive Web App)
- **Real-time Assessment**: Instant risk scoring with visual feedback
- **Multi-Model Display**: Tabbed interface showing results from each model
- **Visual Risk Meter**: Intuitive risk score visualization
- **Responsive Design**: Works on desktop and mobile devices
- **Model Comparison**: Side-by-side performance comparison
- **Export Functionality**: Download assessment data

## 📋 Setup Instructions

### 1. Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Place your credit_risk_data.csv file in the same directory
# Run the backend server
python app.py
```

The backend will start on `http://localhost:5000`

### 2. Frontend Setup

Simply open the HTML file in a web browser or serve it using:

```bash
# Using Python's built-in server
python -m http.server 8080

# Then open http://localhost:8080
```

### 3. CSV Data Format

Your `credit_risk_data.csv` should have these columns:
- `total_transactions`
- `avg_transaction_amount`
- `payment_consistency_score`
- `business_age_months` 
- `digital_footprint_score`

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | System status and health check |
| `/api/train` | POST | Train all AI models |
| `/api/assess` | POST | Assess credit risk using all models |
| `/api/models/performance` | GET | Get model performance metrics |
| `/api/history` | GET | Get assessment history |
| `/api/export/csv` | GET | Export assessments as CSV |

## 🤖 AI Models Used

1. **Neural Network Ensemble** - Deep learning with dual architecture
2. **Random Forest** - Tree-based ensemble method
3. **Gradient Boosting** - Sequential ensemble learning
4. **Logistic Regression** - Linear classification
5. **Support Vector Machine** - Kernel-based classification

## 💡 Usage Examples

### Sample Input Data:
```json
{
    "total_transactions": 18000,
    "avg_transaction_amount": 850,
    "payment_consistency_score": 94,
    "business_age_months": 54,
    "digital_footprint_score": 88
}
```

### Expected Output:
- **Risk Scores** from each model (0-100%)
- **Loan Recommendations** with amount, interest rate, terms
- **Model Performance** metrics and accuracy scores
- **Ensemble Prediction** combining all models

## 🔧 Customization

### Adding New Models:
1. Add model in `MultiModelCreditAssessment.train_all_models()`
2. Update prediction logic in `predict_with_all_models()`
3. Frontend will automatically display new model tabs

### Modifying Risk Criteria:
- Update `generate_loan_recommendation()` method
- Adjust risk thresholds and loan amounts
- Modify interest rates and terms

### Styling Changes:
- Edit CSS in the HTML file
- Customize colors, animations, and layout
- Add new UI components as needed

## 📊 Data Export Features

- **CSV Export**: Download all assessments with model predictions
- **Assessment History**: View recent assessments with timestamps  
- **Model Comparison**: Export performance metrics across models

## 🛠️ Troubleshooting

### Common Issues:

1. **Backend Not Starting**: Check if all requirements are installed
2. **CORS Errors**: Ensure Flask-CORS is installed and configured
3. **Model Training Fails**: Verify CSV file format and data quality
4. **Frontend Connection Issues**: Check if backend is running on port 5000

### Debug Mode:
- Backend runs in debug mode by default
- Check console logs for detailed error messages
- Use browser dev tools to inspect API responses

## 📈 Performance Optimization

- Models are trained once and cached in memory
- Predictions use pre-trained models for speed
- Ensemble predictions combine multiple models efficiently
- Frontend caches results to avoid repeated API calls

## 🔒 Security Considerations

- Add authentication for production use
- Implement rate limiting on API endpoints
- Validate and sanitize all input data
- Use HTTPS in production environment

## 📱 Mobile Responsiveness

The frontend is fully responsive and works on:
- Desktop computers
- Tablets  
- Mobile phones
- Various screen sizes and orientations

## 🌟 Future Enhancements

- Real-time model retraining
- Advanced analytics dashboard
- Integration with banking APIs
- Mobile app development
- Advanced risk visualization
- Automated report generation
