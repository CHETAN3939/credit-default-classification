import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_split_data(filepath):

    df = pd.read_excel(filepath, header=1)
    df.rename(columns={"default payment next month": "credit_default"}, inplace=True)
    df.drop("ID", axis=1, inplace=True)

    features = df.drop("credit_default", axis=1)
    target = df["credit_default"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler
