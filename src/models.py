from sklearn.ensemble import RandomForestClassifier
def train_rf(X, y, **kwargs):
    model = RandomForestClassifier(random_state=42, **kwargs)
    model.fit(X, y)
    return model
