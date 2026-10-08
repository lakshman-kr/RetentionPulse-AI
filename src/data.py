import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_customer_data(n_samples=2500, random_state=42):
    """
    Generates a realistic enterprise SaaS customer churn dataset.
    Features: Tenancy, Monthly Spend, Support Tickets, Usage Drop, Contract Type.
    """
    np.random.seed(random_state)
    
    tenure_months = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.random.uniform(29.0, 499.0, size=n_samples)
    support_tickets = np.random.poisson(lam=2.2, size=n_samples)
    usage_drop_pct = np.random.uniform(-0.4, 0.8, size=n_samples)
    contract_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2]) # 0: Month-to-Month, 1: 1-Yr, 2: 2-Yr
    
    # Latent probability of churn based on domain mechanics
    log_odds = (
        -1.5
        - 0.04 * tenure_months
        + 0.003 * monthly_charges
        + 0.55 * support_tickets
        + 2.8 * usage_drop_pct
        - 1.2 * contract_type
    )
    prob = 1 / (1 + np.exp(-log_odds))
    churn = (np.random.rand(n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'tenure_months': tenure_months,
        'monthly_charges': np.round(monthly_charges, 2),
        'support_tickets': support_tickets,
        'usage_drop_pct': np.round(usage_drop_pct, 2),
        'contract_type': contract_type,
        'churn': churn
    })
    return df

def get_preprocessed_data():
    """
    Performs stratified train/test split and StandardScaler feature scaling.
    Guarantees zero data leakage by fitting scaler strictly on train set.
    """
    df = generate_customer_data()
    X = df.drop(columns=['churn'])
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()

if __name__ == "__main__":
    df = generate_customer_data()
    print(f"Data pipeline ready. Total samples: {len(df)}, Churn baseline: {df['churn'].mean():.2%}")
