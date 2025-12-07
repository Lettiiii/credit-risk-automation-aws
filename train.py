# training_code/train.py
import os
import pandas as pd
import xgboost as xgb

def find_csv(path):
    if os.path.isfile(path) and path.lower().endswith('.csv'):
        return path
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.lower().endswith('.csv'):
                return os.path.join(path, f)
    return None

if __name__ == '__main__':
    # SageMaker environment variables (100% reliable)
    train_dir   = os.environ['SM_CHANNEL_TRAIN']
    model_dir   = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    num_round   = int(os.environ.get('SM_HP_NUM_ROUND', '100'))
    objective   = os.environ.get('SM_HP_OBJECTIVE', 'binary:logistic')
    eval_metric = os.environ.get('SM_HP_EVAL_METRIC', 'auc')

    print(f"SM_CHANNEL_TRAIN = {train_dir}")

    csv_path = find_csv(train_dir)
    if not csv_path:
        raise RuntimeError(f"No CSV file found in training channel: {train_dir}")

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    if 'loan_status' not in df.columns:
        raise RuntimeError("Column 'loan_status' not found in the dataset!")

    print(f"Dataset shape: {df.shape}, Target distribution:\n{df['loan_status'].value_counts()}")

    X = pd.get_dummies(df.drop('loan_status', axis=1), drop_first=True)
    y = df['loan_status'].map({'Fully Paid': 0, 'Charged Off': 1}).astype(int)

    dtrain = xgb.DMatrix(X, label=y)

    params = {
        'objective': objective,
        'eval_metric': eval_metric
    }

    print(f"Training XGBoost with {num_round} rounds...")
    booster = xgb.train(params, dtrain, num_boost_round=num_round)

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'xgboost-model.json')
    booster.save_model(model_path)

    print(f"Training completed successfully!")
    print(f"Model saved to {model_path}")