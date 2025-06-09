import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def create_Xs_and_ys(datasets, scores, val_test_splits=[0.2, 0.1], random_state=42):
    """

    """
    X = np.array([d[0] for d in datasets])
    y = np.array(scores)
    valid_size, test_size = val_test_splits

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=valid_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def sklearn_evaluate_reg(model, X, y, silent=False, desc='Set'):
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    if not silent:
        print(f"{desc} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return mse, mae, r2


def train_and_evaluate_rf(X_train, y_train, X_val, y_val, X_test, y_test, **rf_kwargs):
    """

    """
    rf = RandomForestRegressor(**rf_kwargs)
    rf.fit(X_train, y_train)
    print("Evaluation on Training Set:")
    sklearn_evaluate_reg(rf, X_train, y_train, desc="Train")
    print("Evaluation on Validation Set:")
    sklearn_evaluate_reg(rf, X_val, y_val, desc="Valid")
    print("Evaluation on Test Set:")
    sklearn_evaluate_reg(rf, X_test, y_test, desc="Test")
    return rf

