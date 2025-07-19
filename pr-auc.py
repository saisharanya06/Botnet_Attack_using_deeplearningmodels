from sklearn.metrics import precision_recall_curve, auc

# PR-AUC curve plotting function
def plot_pr_auc(model, name):
    y_score = model.predict(X_test)
    precision, recall, _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.2f})')
