import matplotlib.pyplot as plt

def plot_predictions(y_true, y_pred, title="Model"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.2)
    
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        'r--'
    )
    
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_residuals(y_true, y_pred, title="Residuals"):
    res = y_true - y_pred

    plt.figure()
    plt.scatter(y_true, res, alpha=0.2)
    plt.axhline(0)
    
    plt.xlabel("True RUL")
    plt.ylabel("Residual")
    plt.title(title)
    plt.show()