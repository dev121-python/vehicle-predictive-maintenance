from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


def train_linear(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    model = DecisionTreeRegressor(max_depth=6)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_gradient_boost(X_train, y_train):
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.07,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model