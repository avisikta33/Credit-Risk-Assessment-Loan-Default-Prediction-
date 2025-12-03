"""
Model Training Module for Credit Risk Assessment
Trains and compares multiple ML models
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            roc_curve, precision_recall_curve, average_precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


class CreditRiskModeler:
    """
    Train and evaluate multiple models for credit risk assessment
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def split_data(self, X, y, test_size=0.2, balance=True):
        """
        Split data into train and test sets
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target variable
        test_size : float
            Proportion of test set
        balance : bool
            Whether to apply SMOTE for balancing
            
        Returns:
        --------
        X_train, X_test, y_train, y_test
        """
        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        if balance:
            print("\nApplying SMOTE for class balancing...")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE: {X_train.shape[0]} samples")
            print(f"  Class 0: {(y_train==0).sum()}")
            print(f"  Class 1: {(y_train==1).sum()}")
        
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """
        Initialize all models with default parameters
        """
        print("\nInitializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=self.random_state
            ),
            
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True
            )
        }
        
        print(f"Initialized {len(self.models)} models")
    
    def train_models(self, X_train, y_train):
        """
        Train all models
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        """
        print("\n" + "="*50)
        print("TRAINING MODELS")
        print("="*50)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            try:
                model.fit(X_train, y_train)
                print(f"✓ {name} trained successfully")
            except Exception as e:
                print(f"✗ Error training {name}: {str(e)}")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
            
        Returns:
        --------
        results_df : pd.DataFrame
            Evaluation metrics for all models
        """
        print("\n" + "="*50)
        print("EVALUATING MODELS")
        print("="*50)
        
        results = []
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'Avg Precision': avg_precision
            })
            
            # Store predictions
            self.results[name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'metrics': results[-1]
            }
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  ROC-AUC: {roc_auc:.4f}")
        
        results_df = pd.DataFrame(results)
        
        # Identify best model based on F1-Score
        best_idx = results_df['F1-Score'].idxmax()
        self.best_model_name = results_df.loc[best_idx, 'Model']
        self.best_model = self.models[self.best_model_name]
        
        print("\n" + "="*50)
        print(f"BEST MODEL: {self.best_model_name}")
        print("="*50)
        
        return results_df
    
    def tune_best_model(self, X_train, y_train, X_test, y_test):
        """
        Hyperparameter tuning for the best model
        
        Parameters:
        -----------
        X_train, y_train : Training data
        X_test, y_test : Test data
            
        Returns:
        --------
        best_model : Tuned model
        """
        print(f"\n{'='*50}")
        print(f"TUNING {self.best_model_name}")
        print("="*50)
        
        if self.best_model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
        elif self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, 20],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            }
        elif self.best_model_name == 'Logistic Regression':
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            }
        else:
            print("No tuning defined for this model")
            return self.best_model
        
        print("Starting GridSearchCV...")
        grid_search = GridSearchCV(
            self.best_model,
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")
        
        # Evaluate tuned model
        tuned_model = grid_search.best_estimator_
        y_pred = tuned_model.predict(X_test)
        y_pred_proba = tuned_model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        print(f"\nTuned model performance on test set:")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"  F1-Score: {f1_score(y_test, y_pred):.4f}")
        print(f"  ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        self.best_model = tuned_model
        return tuned_model
    
    def plot_model_comparison(self, results_df, save_path='reports/figures/model_comparison.png'):
        """
        Plot model comparison
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            results_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=False, color='steelblue')
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric, fontsize=12)
            ax.set_xlabel('')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for container in ax.containers:
                ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nModel comparison plot saved to {save_path}")
        plt.close()
    
    def plot_confusion_matrices(self, X_test, y_test, save_path='reports/figures/confusion_matrices.png'):
        """
        Plot confusion matrices for all models
        """
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'})
            axes[idx].set_title(f'{name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Actual', fontsize=10)
            axes[idx].set_xlabel('Predicted', fontsize=10)
        
        # Hide extra subplot
        if n_models < 6:
            axes[5].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to {save_path}")
        plt.close()
    
    def plot_roc_curves(self, X_test, y_test, save_path='reports/figures/roc_curves.png'):
        """
        Plot ROC curves for all models
        """
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
        plt.close()
    
    def save_best_model(self, filepath='models/saved_models/best_model.pkl'):
        """
        Save the best model to disk
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"\nBest model ({self.best_model_name}) saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a saved model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model


def main():
    """
    Main training pipeline
    """
    print("\n" + "="*70)
    print("CREDIT RISK ASSESSMENT - MODEL TRAINING PIPELINE")
    print("="*70)
    
    # Load data
    print("\nLoading preprocessed data...")
    X = pd.read_csv('data/processed/X_engineered.csv')
    y = pd.read_csv('data/processed/y_processed.csv').squeeze()
    
    # Initialize modeler
    modeler = CreditRiskModeler(random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = modeler.split_data(X, y, test_size=0.2, balance=True)
    
    # Initialize and train models
    modeler.initialize_models()
    modeler.train_models(X_train, y_train)
    
    # Evaluate models
    results_df = modeler.evaluate_models(X_test, y_test)
    
    # Save results
    results_df.to_csv('reports/model_comparison_results.csv', index=False)
    print("\nModel comparison results saved to reports/model_comparison_results.csv")
    
    # Create visualizations
    modeler.plot_model_comparison(results_df)
    modeler.plot_confusion_matrices(X_test, y_test)
    modeler.plot_roc_curves(X_test, y_test)
    
    # Tune best model
    tuned_model = modeler.tune_best_model(X_train, y_train, X_test, y_test)
    
    # Save best model
    modeler.save_best_model()
    
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()
