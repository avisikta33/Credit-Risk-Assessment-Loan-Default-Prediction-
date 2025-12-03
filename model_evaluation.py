"""
Model Evaluation and Business Impact Analysis
Provides detailed evaluation metrics and business insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, precision_recall_curve,
                            average_precision_score, accuracy_score)
import shap
import pickle
import warnings
warnings.filterwarnings('ignore')


class BusinessImpactAnalyzer:
    """
    Analyze business impact of credit risk model
    """
    
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.y_pred_proba = model.predict_proba(X_test)[:, 1]
        
    def calculate_cost_benefit(self, default_cost=10000, loss_per_default=5000, 
                               revenue_per_good_loan=1000):
        """
        Calculate cost-benefit analysis
        
        Parameters:
        -----------
        default_cost : float
            Cost of approving a loan that defaults
        loss_per_default : float
            Average loss per default
        revenue_per_good_loan : float
            Revenue from a good loan
            
        Returns:
        --------
        dict : Cost-benefit metrics
        """
        print("\n" + "="*50)
        print("COST-BENEFIT ANALYSIS")
        print("="*50)
        
        cm = confusion_matrix(self.y_test, self.y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate costs and revenues
        cost_fp = fp * default_cost  # False positives: wrongly rejected good loans
        cost_fn = fn * loss_per_default  # False negatives: approved bad loans
        revenue_tn = tn * revenue_per_good_loan  # Correctly approved good loans
        saved_tp = tp * loss_per_default  # Correctly rejected bad loans
        
        total_cost = cost_fp + cost_fn
        total_revenue = revenue_tn + saved_tp
        net_profit = total_revenue - total_cost
        
        # Baseline (approve all)
        total_loans = len(self.y_test)
        total_defaults = self.y_test.sum()
        total_good_loans = total_loans - total_defaults
        
        baseline_revenue = total_good_loans * revenue_per_good_loan
        baseline_cost = total_defaults * loss_per_default
        baseline_profit = baseline_revenue - baseline_cost
        
        improvement = net_profit - baseline_profit
        improvement_pct = (improvement / abs(baseline_profit)) * 100
        
        results = {
            'True Negatives (Correct Approvals)': tn,
            'False Positives (Wrong Rejections)': fp,
            'False Negatives (Wrong Approvals)': fn,
            'True Positives (Correct Rejections)': tp,
            'Cost of False Positives': f'${cost_fp:,.2f}',
            'Cost of False Negatives': f'${cost_fn:,.2f}',
            'Revenue from True Negatives': f'${revenue_tn:,.2f}',
            'Savings from True Positives': f'${saved_tp:,.2f}',
            'Total Cost': f'${total_cost:,.2f}',
            'Total Revenue': f'${total_revenue:,.2f}',
            'Net Profit': f'${net_profit:,.2f}',
            'Baseline Profit (Approve All)': f'${baseline_profit:,.2f}',
            'Improvement over Baseline': f'${improvement:,.2f} ({improvement_pct:.2f}%)'
        }
        
        print("\nConfusion Matrix Breakdown:")
        for key, value in list(results.items())[:4]:
            print(f"  {key}: {value}")
        
        print("\nFinancial Impact:")
        for key, value in list(results.items())[4:]:
            print(f"  {key}: {value}")
        
        return results
    
    def calculate_risk_scores(self):
        """
        Calculate risk scores for all loans
        
        Returns:
        --------
        pd.DataFrame : Risk scores and categories
        """
        risk_scores = self.y_pred_proba * 100
        
        # Categorize risk
        risk_categories = pd.cut(
            risk_scores,
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        risk_df = pd.DataFrame({
            'Risk_Score': risk_scores,
            'Risk_Category': risk_categories,
            'Actual_Default': self.y_test.values,
            'Predicted_Default': self.y_pred
        })
        
        print("\n" + "="*50)
        print("RISK SCORE DISTRIBUTION")
        print("="*50)
        print(risk_df['Risk_Category'].value_counts().sort_index())
        
        print("\nDefault Rate by Risk Category:")
        default_by_category = risk_df.groupby('Risk_Category')['Actual_Default'].agg(['mean', 'count'])
        default_by_category['mean'] = default_by_category['mean'] * 100
        default_by_category.columns = ['Default Rate (%)', 'Count']
        print(default_by_category)
        
        return risk_df
    
    def optimal_threshold_analysis(self):
        """
        Find optimal classification threshold
        
        Returns:
        --------
        float : Optimal threshold
        """
        print("\n" + "="*50)
        print("OPTIMAL THRESHOLD ANALYSIS")
        print("="*50)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(self.y_test, self.y_pred_proba)
        
        # F1 scores for different thresholds
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Find optimal threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
        print(f"  Precision: {precision[optimal_idx]:.4f}")
        print(f"  Recall: {recall[optimal_idx]:.4f}")
        print(f"  F1-Score: {f1_scores[optimal_idx]:.4f}")
        
        # Compare with default threshold (0.5)
        y_pred_default = (self.y_pred_proba >= 0.5).astype(int)
        y_pred_optimal = (self.y_pred_proba >= optimal_threshold).astype(int)
        
        from sklearn.metrics import f1_score
        print(f"\nDefault Threshold (0.5) F1-Score: {f1_score(self.y_test, y_pred_default):.4f}")
        print(f"Optimal Threshold F1-Score: {f1_score(self.y_test, y_pred_optimal):.4f}")
        
        return optimal_threshold
    
    def approval_rate_analysis(self):
        """
        Analyze approval rates and default rates at different thresholds
        """
        print("\n" + "="*50)
        print("APPROVAL RATE ANALYSIS")
        print("="*50)
        
        thresholds = np.arange(0.3, 0.8, 0.05)
        results = []
        
        for thresh in thresholds:
            y_pred_thresh = (self.y_pred_proba < thresh).astype(int)  # 1 = approve
            
            approved_mask = (y_pred_thresh == 1)
            approval_rate = approved_mask.sum() / len(y_pred_thresh)
            
            if approved_mask.sum() > 0:
                default_rate = self.y_test[approved_mask].sum() / approved_mask.sum()
            else:
                default_rate = 0
            
            results.append({
                'Threshold': thresh,
                'Approval_Rate': approval_rate * 100,
                'Default_Rate': default_rate * 100
            })
        
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        return results_df
    
    def plot_business_metrics(self, save_path='reports/figures/business_metrics.png'):
        """
        Plot business-relevant metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Risk Score Distribution
        risk_df = self.calculate_risk_scores()
        risk_counts = risk_df['Risk_Category'].value_counts().sort_index()
        axes[0, 0].bar(range(len(risk_counts)), risk_counts.values, color='steelblue')
        axes[0, 0].set_xticks(range(len(risk_counts)))
        axes[0, 0].set_xticklabels(risk_counts.index, rotation=45)
        axes[0, 0].set_title('Risk Score Distribution', fontweight='bold')
        axes[0, 0].set_ylabel('Number of Loans')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Default Rate by Risk Category
        default_by_risk = risk_df.groupby('Risk_Category')['Actual_Default'].mean() * 100
        axes[0, 1].bar(range(len(default_by_risk)), default_by_risk.values, color='coral')
        axes[0, 1].set_xticks(range(len(default_by_risk)))
        axes[0, 1].set_xticklabels(default_by_risk.index, rotation=45)
        axes[0, 1].set_title('Default Rate by Risk Category', fontweight='bold')
        axes[0, 1].set_ylabel('Default Rate (%)')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Precision-Recall Trade-off
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        axes[1, 0].plot(recall, precision, linewidth=2, color='green')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve', fontweight='bold')
        axes[1, 0].grid(alpha=0.3)
        
        # 4. Approval vs Default Rate
        approval_df = self.approval_rate_analysis()
        ax2 = axes[1, 1].twinx()
        axes[1, 1].plot(approval_df['Threshold'], approval_df['Approval_Rate'], 
                       'b-', linewidth=2, label='Approval Rate')
        ax2.plot(approval_df['Threshold'], approval_df['Default_Rate'], 
                'r-', linewidth=2, label='Default Rate')
        axes[1, 1].set_xlabel('Decision Threshold')
        axes[1, 1].set_ylabel('Approval Rate (%)', color='b')
        ax2.set_ylabel('Default Rate (%)', color='r')
        axes[1, 1].set_title('Approval vs Default Rate by Threshold', fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nBusiness metrics plot saved to {save_path}")
        plt.close()
    
    def generate_shap_analysis(self, save_path='reports/figures/shap_summary.png'):
        """
        Generate SHAP values for model interpretability
        """
        print("\n" + "="*50)
        print("SHAP ANALYSIS (Model Interpretability)")
        print("="*50)
        
        try:
            # Sample data for faster computation
            sample_size = min(1000, len(self.X_test))
            X_sample = self.X_test.sample(n=sample_size, random_state=42)
            
            print(f"\nCalculating SHAP values for {sample_size} samples...")
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
            
            # If binary classification, take positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP summary plot saved to {save_path}")
            plt.close()
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': X_sample.columns,
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))
            
            return shap_values, feature_importance
            
        except Exception as e:
            print(f"Error in SHAP analysis: {str(e)}")
            return None, None


def main():
    """
    Main evaluation pipeline
    """
    print("\n" + "="*70)
    print("CREDIT RISK MODEL - EVALUATION & BUSINESS IMPACT ANALYSIS")
    print("="*70)
    
    # Load data and model
    print("\nLoading test data and model...")
    X_test = pd.read_csv('data/processed/X_engineered.csv')
    y_test = pd.read_csv('data/processed/y_processed.csv').squeeze()
    
    with open('models/saved_models/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Split to get test set (use same random state as training)
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        X_test, y_test, test_size=0.2, random_state=42, stratify=y_test
    )
    
    # Initialize analyzer
    analyzer = BusinessImpactAnalyzer(model, X_test, y_test)
    
    # Cost-benefit analysis
    cost_benefit = analyzer.calculate_cost_benefit()
    
    # Risk scores
    risk_df = analyzer.calculate_risk_scores()
    risk_df.to_csv('reports/risk_scores.csv', index=False)
    
    # Optimal threshold
    optimal_threshold = analyzer.optimal_threshold_analysis()
    
    # Business metrics visualization
    analyzer.plot_business_metrics()
    
    # SHAP analysis
    shap_values, feature_importance = analyzer.generate_shap_analysis()
    if feature_importance is not None:
        feature_importance.to_csv('reports/feature_importance.csv', index=False)
    
    # Generate detailed report
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  - reports/risk_scores.csv")
    print("  - reports/feature_importance.csv")
    print("  - reports/figures/business_metrics.png")
    print("  - reports/figures/shap_summary.png")


if __name__ == "__main__":
    main()
