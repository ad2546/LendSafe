"""
Process Lending Club dataset for LendSafe
Creates cleaned dataset ready for model training
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

def create_synthetic_lending_data(n_samples=1000):
    """
    Create synthetic lending data for demo purposes
    In production, use actual Lending Club dataset
    """
    print(f"📊 Creating {n_samples} synthetic loan applications...")

    np.random.seed(42)

    # Generate features
    data = {
        'loan_amount': np.random.randint(5000, 50000, n_samples),
        'credit_score': np.random.randint(550, 850, n_samples),
        'annual_income': np.random.randint(30000, 200000, n_samples),
        'dti_ratio': np.random.uniform(5, 45, n_samples),
        'employment_length': np.random.randint(0, 30, n_samples),
        'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE'], n_samples),
        'loan_purpose': np.random.choice([
            'debt_consolidation', 'credit_card', 'home_improvement',
            'major_purchase', 'small_business', 'car'
        ], n_samples),
        'delinq_2yrs': np.random.poisson(0.3, n_samples),
        'inquiries_6mon': np.random.poisson(0.5, n_samples),
        'open_accounts': np.random.randint(2, 20, n_samples),
        'revolving_util': np.random.uniform(0, 100, n_samples),
    }

    df = pd.DataFrame(data)

    # Create approval target based on rules
    df['approved'] = (
        (df['credit_score'] >= 640) &
        (df['dti_ratio'] <= 43) &
        (df['delinq_2yrs'] <= 1) &
        (df['annual_income'] >= 25000)
    ).astype(int)

    # Add some noise
    flip_mask = np.random.random(n_samples) < 0.05  # 5% random flips
    df.loc[flip_mask, 'approved'] = 1 - df.loc[flip_mask, 'approved']

    # Calculate approval rate
    approval_rate = df['approved'].mean() * 100
    print(f"✅ Generated {n_samples} applications")
    print(f"📈 Approval rate: {approval_rate:.1f}%")

    return df


def engineer_features(df):
    """Create additional features for modeling"""
    print("\n🔧 Engineering features...")

    # Income to loan ratio
    df['income_to_loan'] = df['annual_income'] / df['loan_amount']

    # Credit utilization buckets
    df['util_bucket'] = pd.cut(
        df['revolving_util'],
        bins=[0, 30, 60, 90, 100],
        labels=['low', 'medium', 'high', 'very_high']
    )

    # Risk score (simple composite)
    df['risk_score'] = (
        (df['credit_score'] / 10) +
        ((50 - df['dti_ratio']) * 2) +
        (df['open_accounts'] * 2) -
        (df['delinq_2yrs'] * 20) -
        (df['inquiries_6mon'] * 10)
    )

    # Encode categorical
    df = pd.get_dummies(df, columns=['home_ownership', 'loan_purpose', 'util_bucket'])

    print(f"✅ Created {len(df.columns)} total features")

    return df


def split_and_save_data(df):
    """Split into train/validation/test sets"""
    print("\n📂 Splitting and saving data...")

    # Features for modeling
    feature_cols = [col for col in df.columns if col != 'approved']
    X = df[feature_cols]
    y = df['approved']

    # Split: 70% train, 15% val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    # Save splits
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pd.concat([X_train, y_train], axis=1).to_csv(
        PROCESSED_DIR / 'train.csv', index=False
    )
    pd.concat([X_val, y_val], axis=1).to_csv(
        PROCESSED_DIR / 'val.csv', index=False
    )
    pd.concat([X_test, y_test], axis=1).to_csv(
        PROCESSED_DIR / 'test.csv', index=False
    )

    # Also save full dataset
    df.to_csv(PROCESSED_DIR / 'full_data.csv', index=False)

    print(f"✅ Train set: {len(X_train)} samples")
    print(f"✅ Val set: {len(X_val)} samples")
    print(f"✅ Test set: {len(X_test)} samples")
    print(f"📁 Saved to: {PROCESSED_DIR}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """Main processing pipeline"""
    print("="*60)
    print("LENDING DATA PROCESSING PIPELINE")
    print("="*60)

    # Check for existing Lending Club data
    lending_club_file = RAW_DIR / "lending_club.csv"

    if lending_club_file.exists():
        print(f"\n📥 Loading Lending Club dataset from {lending_club_file}")
        df = pd.read_csv(lending_club_file)
    else:
        print("\n⚠️  Lending Club dataset not found")
        print("💡 Creating synthetic data for demo...")
        df = create_synthetic_lending_data(n_samples=5000)

    # Engineer features
    df = engineer_features(df)

    # Split and save
    split_and_save_data(df)

    print("\n" + "="*60)
    print("✅ DATA PROCESSING COMPLETE!")
    print("="*60)

    # Print feature summary
    print("\n📊 Feature Summary:")
    print(f"Total features: {len([c for c in df.columns if c != 'approved'])}")
    print(f"Approval rate: {df['approved'].mean()*100:.1f}%")
    print(f"\nKey statistics:")
    print(df[['loan_amount', 'credit_score', 'annual_income', 'dti_ratio']].describe())


if __name__ == "__main__":
    main()
