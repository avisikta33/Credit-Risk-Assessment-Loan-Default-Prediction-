# 🏦 Credit Risk Assessment & Loan Default Prediction

## 📊 Project Overview

A comprehensive machine learning project for predicting loan defaults and assessing credit risk. This project demonstrates end-to-end ML pipeline development with a focus on business impact analysis, making it ideal for Business Analyst portfolios.

### 🎯 Business Objectives
- **Minimize loan defaults** by identifying high-risk applicants
- **Optimize approval rates** while maintaining acceptable risk levels
- **Quantify financial impact** through cost-benefit analysis
- **Provide explainable predictions** for regulatory compliance

### 🔑 Key Features
- ✅ Multiple ML algorithms (Logistic Regression, Random Forest, XGBoost, Neural Networks)
- ✅ Advanced feature engineering (40+ engineered features)
- ✅ SHAP-based model interpretability
- ✅ Cost-benefit analysis with business metrics
- ✅ Risk scoring and segmentation
- ✅ Interactive visualizations and reports

---

## 📁 Project Structure

```
credit-risk-assessment/
│
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and engineered data
│   └── external/               # External data sources
│
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Model_Interpretation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py   # Data cleaning module
│   ├── feature_engineering.py  # Feature creation
│   ├── model_training.py       # Model training pipeline
│   ├── model_evaluation.py     # Evaluation & business impact
│   └── utils.py                # Helper functions
│
├── models/
│   ├── saved_models/           # Trained model files (.pkl)
│   └── model_configs/          # Hyperparameter configurations
│
├── reports/
│   ├── figures/                # Plots and visualizations
│   ├── model_comparison_results.csv
│   ├── risk_scores.csv
│   └── business_impact_report.pdf
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_features.py
│
├── requirements.txt            # Python dependencies
├── config.yaml                 # Configuration file
├── README.md                   # This file
└── .gitignore
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB RAM minimum (16GB recommended)

### Installation

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd credit-risk-assessment
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download dataset**

Download the Lending Club Loan Data from Kaggle:
- Dataset: [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- Place the CSV file in `data/raw/lending_club_loans.csv`

Alternative datasets:
- [German Credit Data](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)

---

## 📖 Usage

### Option 1: Run Complete Pipeline

```bash
# Step 1: Data Preprocessing
python src/data_preprocessing.py

# Step 2: Feature Engineering
python src/feature_engineering.py

# Step 3: Model Training
python src/model_training.py

# Step 4: Model Evaluation
python src/model_evaluation.py
```

### Option 2: Use Jupyter Notebooks

```bash
jupyter notebook
```

Then run notebooks in sequence:
1. `01_EDA.ipynb` - Understand the data
2. `02_Feature_Engineering.ipynb` - Create features
3. `03_Model_Training.ipynb` - Train models
4. `04_Model_Interpretation.ipynb` - Interpret results

---

## 🧪 Model Performance

### Trained Models
1. **Logistic Regression** - Baseline interpretable model
2. **Random Forest** - Ensemble method with feature importance
3. **XGBoost** - Gradient boosting (typically best performer)
4. **Gradient Boosting** - Alternative boosting algorithm
5. **Neural Network** - Deep learning approach

### Expected Performance Metrics
| Metric | Target | Typical Result |
|--------|---------|----------------|
| **Accuracy** | >85% | 87-92% |
| **Precision** | >80% | 82-88% |
| **Recall** | >75% | 77-85% |
| **F1-Score** | >78% | 80-86% |
| **ROC-AUC** | >0.85 | 0.88-0.93 |

### Business Impact (Example)
```
Assumptions:
- Average loan amount: $15,000
- Default cost: $10,000 per defaulted loan
- Revenue per good loan: $1,000
- Test set: 10,000 loans

Results:
✅ Correctly approved: 8,200 loans → Revenue: $8,200,000
✅ Correctly rejected: 1,500 defaults → Savings: $15,000,000
❌ Wrongly rejected: 200 loans → Lost revenue: $200,000
❌ Wrongly approved: 100 defaults → Cost: $1,000,000

Net Profit: $22,000,000
Improvement over baseline (approve all): +$8,500,000 (63% improvement)
```

---

## 🔬 Feature Engineering

### Created Features (40+ features)

**Credit Utilization Features:**
- `credit_limit` - Calculated credit limit
- `debt_to_income_ratio` - Monthly debt burden
- `loan_to_income` - Loan amount relative to income
- `acc_closure_rate` - Account closure rate

**FICO Score Features:**
- `fico_score_avg` - Average FICO score
- `fico_score_range` - FICO score range
- `fico_category` - FICO risk category (Poor to Excellent)

**Loan-Specific Features:**
- `int_rate_category` - Interest rate risk level
- `loan_amnt_category` - Loan size category
- `funded_ratio` - Funding completion ratio

**Derogatory Record Features:**
- `total_negative_records` - Sum of all negative records
- `has_negative_record` - Binary flag for any negative record
- `is_delinquent` - Delinquency indicator
- `has_bankruptcy` - Bankruptcy flag

**Interaction Features:**
- `fico_dti_interaction` - FICO × DTI
- `income_loan_interaction` - Income × Loan Amount
- `int_loan_interaction` - Interest Rate × Loan Amount

---

## 📊 Model Interpretation

### SHAP (SHapley Additive exPlanations)

The project uses SHAP values for model interpretability:

- **Global Explanations**: Feature importance across all predictions
- **Local Explanations**: Individual prediction explanations
- **Force Plots**: Visualize how features push predictions

### Top Predictive Features (Typical)
1. **FICO Score** - Credit history quality
2. **Debt-to-Income Ratio** - Debt burden
3. **Interest Rate** - Loan cost indicator
4. **Annual Income** - Ability to repay
5. **Loan Amount** - Exposure level
6. **Derogatory Records** - Past defaults/delinquencies
7. **Employment Length** - Income stability
8. **Revolving Utilization** - Current credit usage

---

## 💼 Business Analysis Features

### 1. Risk Scoring System
- **Very Low Risk** (0-20%): Auto-approve candidates
- **Low Risk** (20-40%): Standard approval process
- **Medium Risk** (40-60%): Enhanced review required
- **High Risk** (60-80%): Reject or require collateral
- **Very High Risk** (80-100%): Auto-reject

### 2. Cost-Benefit Analysis
Quantifies financial impact:
- Revenue from correctly approved loans
- Savings from correctly rejected applications
- Cost of false positives (lost revenue)
- Cost of false negatives (defaults)

### 3. Threshold Optimization
Finds optimal decision threshold by:
- Maximizing F1-Score
- Balancing precision and recall
- Considering business constraints

### 4. Portfolio Analysis
- Default rate by risk category
- Approval rate vs default rate trade-offs
- Expected loss calculations

---

## 📈 Visualizations Generated

The project generates comprehensive visualizations:

1. **Model Comparison Charts**
   - Accuracy, Precision, Recall, F1-Score comparison
   - ROC curves for all models
   - Confusion matrices

2. **Business Metrics Plots**
   - Risk score distribution
   - Default rate by risk category
   - Precision-recall trade-off
   - Approval vs default rate curves

3. **Feature Analysis**
   - SHAP summary plots
   - Feature importance rankings
   - Feature correlations

4. **EDA Visualizations**
   - Distribution plots
   - Box plots for outliers
   - Correlation heatmaps
   - Class imbalance visualization

---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

### Technical Skills
- ✅ Data preprocessing and cleaning
- ✅ Feature engineering and selection
- ✅ Multiple ML algorithms implementation
- ✅ Hyperparameter tuning with GridSearchCV
- ✅ Handling imbalanced datasets (SMOTE)
- ✅ Model evaluation and comparison
- ✅ Model interpretability (SHAP)

### Business Analysis Skills
- ✅ Cost-benefit analysis
- ✅ Risk assessment and scoring
- ✅ Business metric definition
- ✅ Stakeholder communication
- ✅ Executive reporting
- ✅ ROI calculation

### Software Engineering
- ✅ Modular code structure
- ✅ Version control best practices
- ✅ Documentation
- ✅ Testing and validation
- ✅ Reproducible research

---

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
data:
  raw_path: "data/raw/lending_club_loans.csv"
  processed_path: "data/processed/"
  test_size: 0.2
  random_state: 42

preprocessing:
  missing_threshold: 0.3
  outlier_method: "iqr"
  scaling_method: "standard"

feature_engineering:
  n_features: 50
  feature_selection_method: "mutual_info"
  polynomial_degree: 2

modeling:
  balance_method: "smote"
  cv_folds: 5
  scoring_metric: "f1"

business:
  default_cost: 10000
  loss_per_default: 5000
  revenue_per_good_loan: 1000
```

---

## 🧪 Testing

Run tests to ensure code quality:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test
pytest tests/test_preprocessing.py
```

---

## 📝 Results and Reports

After running the pipeline, find results in:

- `reports/model_comparison_results.csv` - Model performance metrics
- `reports/risk_scores.csv` - Individual loan risk scores
- `reports/feature_importance.csv` - Feature importance rankings
- `reports/figures/` - All visualization plots
- `reports/business_impact_report.pdf` - Executive summary

---

## 🚦 Common Issues and Solutions

### Issue 1: Memory Error
**Solution**: Reduce sample size in SHAP analysis or use smaller dataset

### Issue 2: SMOTE Taking Too Long
**Solution**: Reduce training set size or disable SMOTE in config

### Issue 3: Missing Dataset
**Solution**: Ensure CSV file is in `data/raw/` with correct name

### Issue 4: Module Import Errors
**Solution**: Run `pip install -r requirements.txt` again

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Lending Club** for providing the loan dataset
- **Kaggle** for hosting the data
- **scikit-learn** for ML algorithms
- **SHAP** library for model interpretability
- **XGBoost** team for the excellent boosting algorithm

---

## 📧 Contact

**Your Name** - [Your Email] - [LinkedIn Profile]

Project Link: [https://github.com/yourusername/credit-risk-assessment](https://github.com/yourusername/credit-risk-assessment)

---

## 🌟 Star This Repository

If you find this project helpful for your portfolio or learning, please give it a ⭐!

---

## 📚 Additional Resources

### Related Articles
- [Credit Risk Modeling with Machine Learning](https://example.com)
- [Interpretable Machine Learning for Finance](https://example.com)
- [Business Impact of ML in Banking](https://example.com)

### Further Reading
- "Credit Risk Analytics" by Bart Baesens
- "Machine Learning for Asset Managers" by Marcos López de Prado
- scikit-learn documentation: https://scikit-learn.org/

---

**Built with ❤️ for Business Analyst Portfolio**

*Last Updated: December 2024*
