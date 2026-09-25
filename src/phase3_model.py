import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import joblib

def train_pair_matcher(df_train, df_val, features, target='label', model_path='experiments/models/lgbm_matcher.pkl'):
    """
    Trains a LightGBM model to predict whether a candidate pair is a true match or not.
    """
    print(f"Training LightGBM on {len(df_train)} pairs with {len(features)} features...")
    
    X_train = df_train[features]
    y_train = df_train[target]
    
    X_val = df_val[features]
    y_val = df_val[target]
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        'objective': 'binary',
        'metric': 'auc', # Area Under Curve is great for imbalanced pair matching
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1
    }
    
    print("Starting LightGBM training...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1500,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True), 
            lgb.log_evaluation(100)
        ]
    )
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance (Gain):")
    print(importance.head(15))
    
    return model

def load_and_predict(df, features, model_path='experiments/models/lgbm_matcher.pkl'):
    model = joblib.load(model_path)
    df = df.copy()
    df['match_prob'] = model.predict(df[features])
    return df
