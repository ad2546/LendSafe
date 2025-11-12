"""
Train XGBoost risk scoring model for loan approval
"""
import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import joblib

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
MODEL_DIR = Path(__file__).parent.parent / "models" / "risk_model"

def load_data():
    """Load processed data"""
    print("📥 Loading data...")

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    X_train = train.drop('approved', axis=1)
    y_train = train['approved']

    X_val = val.drop('approved', axis=1)
    y_val = val['approved']

    X_test = test.drop('approved', axis=1)
    y_test = test['approved']

    print(f"✅ Loaded train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgboost_model(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier"""
    print("\n🚀 Training XGBoost model...")

    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'random_state': 42,
        'eval_metric': 'auc'
    }

    # Train model
    model = xgb.XGBClassifier(**params)

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )

    print("✅ Model training complete!")

    return model


def evaluate_model(model, X_test, y_test, X_val=None, y_val=None):
    """Evaluate model performance"""
    print("\n📊 Evaluating model...")

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print("\n" + "="*60)
    print("MODEL PERFORMANCE ON TEST SET")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives:  {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives:  {cm[1,1]}")

    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n" + "="*60)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("="*60)
    print(feature_importance.head(10).to_string(index=False))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'feature_importance': feature_importance
    }


def save_model(model, metrics, feature_importance):
    """Save trained model and artifacts"""
    print("\n💾 Saving model...")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save_model(MODEL_DIR / "xgboost_model.json")
    joblib.dump(model, MODEL_DIR / "xgboost_model.pkl")

    # Save metrics
    pd.DataFrame([metrics]).to_csv(MODEL_DIR / "metrics.csv", index=False)

    # Save feature importance
    feature_importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    print(f"✅ Model saved to: {MODEL_DIR}")


def test_prediction(model, X_test):
    """Test single prediction"""
    print("\n🧪 Testing single prediction...")

    # Get a sample
    sample = X_test.iloc[0:1]

    # Predict
    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0]

    print("\n" + "="*60)
    print("SAMPLE PREDICTION TEST")
    print("="*60)
    print(f"Prediction: {'APPROVED' if prediction == 1 else 'REJECTED'}")
    print(f"Probability of Approval: {probability[1]:.2%}")
    print(f"Probability of Rejection: {probability[0]:.2%}")

    # Show some key features
    print("\nKey Features:")
    for col in ['credit_score', 'dti_ratio', 'loan_amount', 'annual_income']:
        if col in sample.columns:
            print(f"  {col}: {sample[col].values[0]:.2f}")


def main():
    """Main training pipeline"""
    print("="*60)
    print("XGBOOST RISK MODEL TRAINING")
    print("="*60)

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # Train model
    model = train_xgboost_model(X_train, y_train, X_val, y_val)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model, metrics, metrics['feature_importance'])

    # Test single prediction
    test_prediction(model, X_test)

    print("\n" + "="*60)
    print("✅ RISK MODEL TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
