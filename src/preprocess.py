from sklearn.preprocessing import StandardScaler

def scale_data(X_train, X_val):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled, scaler