import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained classification model and returns performance metrics.

    Parameters
    ----------
    model : object
        Trained model with a predict method.
    X_test : pandas.DataFrame
        Test feature set.
    y_test : array-like
        True labels for the test set.

    Returns
    -------
    tuple
        A tuple containing predicted labels and a dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)
    results = {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision_macro": metrics.precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": metrics.recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro": metrics.f1_score(y_test, y_pred, average="macro", zero_division=0),
        "specificity": specificity
    }

    return y_pred, results


def print_metrics(results):
    """
    Prints evaluation metrics to the console.

    Parameters
    ----------
    results : dict
        Dictionary containing evaluation metric values.

    Returns
    -------
    None
    """
    print("Accuracy:", results["accuracy"])
    print("Precision:", results["precision_macro"])
    print("Recall:", results["recall_macro"])
    print("F1 Score:", results["f1_macro"])
    print("Specificity:", results["specificity"])
    print("AUC:", results.get("auc", "N/A"))


def plot_class_distribution(y):
    """
    Plots the class distribution of target labels.

    Parameters
    ----------
    y : pandas.Series or array-like
        Target labels.

    Returns
    -------
    None
    """
    plt.figure(figsize=(8, 5))
    y.value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_test, y_pred, title="Confusion Matrix"):
    """
    Plots a confusion matrix for true versus predicted labels.

    Parameters
    ----------
    y_test : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    title : str, optional
        Plot title. Defaults to "Confusion Matrix".

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_random_forest_feature_importance(model, X, top_n=15):
    """
    Displays the top feature importances from a trained Random Forest model.

    Parameters
    ----------
    model : object
        Trained Random Forest model with feature_importances_.
    X : pandas.DataFrame
        Training or evaluation feature set providing feature names.
    top_n : int, optional
        Number of top features to display. Defaults to 15.

    Returns
    -------
    None
    """
    if not hasattr(model, "feature_importances_"):
        print("This model does not have feature_importances_.")
        return

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    top_features.sort_values().plot(kind="barh")
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def plot_logistic_regression_coefficients(model, X, top_n=15):
    """
    Visualizes the top logistic regression coefficients.

    Parameters
    ----------
    model : object
        Trained pipeline containing a logistic regression classifier.
    X : pandas.DataFrame
        Feature set used to determine coefficient names.
    top_n : int, optional
        Number of coefficients to display. Defaults to 15.

    Returns
    -------
    None
    """
    classifier = model.named_steps["classifier"]

    if not hasattr(classifier, "coef_"):
        print("This model does not have coefficients.")
        return

    coefficients = pd.Series(classifier.coef_[0], index=X.columns)

    top_positive = coefficients.sort_values(ascending=False).head(top_n)
    top_negative = coefficients.sort_values(ascending=True).head(top_n)

    top_coefficients = pd.concat([top_negative, top_positive])

    plt.figure(figsize=(10, 8))
    top_coefficients.sort_values().plot(kind="barh")
    plt.title(f"Top Logistic Regression Coefficients")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.show()


def plot_train_vs_test_accuracy_rf(X_train, X_test, y_train, y_test, random_state=42):
    """
    Plots train and test accuracy across different Random Forest tree counts.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training set features.
    X_test : pandas.DataFrame
        Test set features.
    y_train : array-like
        Training set labels.
    y_test : array-like
        Test set labels.
    random_state : int, optional
        Random seed for classifier training. Defaults to 42.

    Returns
    -------
    None
    """

    n_trees_list = [1, 5, 10, 20, 50, 100, 150, 200]

    train_scores = []
    test_scores = []

    for n in n_trees_list:
        model = RandomForestClassifier(
            n_estimators=n,
            random_state=random_state,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_scores.append(metrics.accuracy_score(y_train, train_pred))
        test_scores.append(metrics.accuracy_score(y_test, test_pred))

    plt.figure(figsize=(8, 5))
    plt.plot(n_trees_list, train_scores, marker="o", label="Train Accuracy")
    plt.plot(n_trees_list, test_scores, marker="o", label="Test Accuracy")
    plt.title("Random Forest Train vs Test Accuracy")
    plt.xlabel("Number of Trees")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
    



def plot_roc_auc(
    model,
    X_test,
    y_test,
    positive_label="anorexia",
    title="ROC Curve"
):
    """
    Plots the ROC curve for a binary classification model and returns the AUC.

    Parameters
    ----------
    model : object
        Trained model with predict_proba support.
    X_test : pandas.DataFrame
        Test features.
    y_test : pandas.Series or array-like
        True labels.
    positive_label : str, optional
        Label considered the positive class. Defaults to "anorexia".
    title : str, optional
        Plot title. Defaults to "ROC Curve".

    Returns
    -------
    float
        Calculated area under the ROC curve.
    """

    if not hasattr(model, "predict_proba"):
        raise ValueError("This model does not support predict_proba().")

    # Get probability matrix
    y_proba = model.predict_proba(X_test)

    # Find the index of the positive class
    class_labels = list(model.classes_) if hasattr(model, "classes_") else list(model.named_steps["classifier"].classes_)

    if positive_label not in class_labels:
        raise ValueError(
            f"Positive label '{positive_label}' not found in model classes: {class_labels}"
        )

    positive_index = class_labels.index(positive_label)

    # Probability of the positive class
    y_score = y_proba[:, positive_index]

    # Convert y_test to binary: anorexia = 1, control = 0
    y_test_binary = (y_test == positive_label).astype(int)

    fpr, tpr, thresholds = roc_curve(y_test_binary, y_score)
    auc_score = roc_auc_score(y_test_binary, y_score)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")

    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Recall")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return auc_score


def calculate_auc(model, X_test, y_test, positive_label="anorexia"):
    """
    Calculates the AUC score for a binary classification model.

    Parameters
    ----------
    model : object
        Trained model with predict_proba support.
    X_test : pandas.DataFrame
        Test set features.
    y_test : pandas.Series or array-like
        True labels.
    positive_label : str, optional
        Label considered positive in binary classification. Defaults to "anorexia".

    Returns
    -------
    float
        The area under the ROC curve.
    """

    if not hasattr(model, "predict_proba"):
        raise ValueError("This model does not support predict_proba().")

    y_proba = model.predict_proba(X_test)

    if hasattr(model, "classes_"):
        class_labels = list(model.classes_)
    else:
        class_labels = list(model.named_steps["classifier"].classes_)

    if positive_label not in class_labels:
        raise ValueError(
            f"Positive label '{positive_label}' not found in model classes: {class_labels}"
        )

    positive_index = class_labels.index(positive_label)
    y_score = y_proba[:, positive_index]

    y_test_binary = (y_test == positive_label).astype(int)

    auc_score = roc_auc_score(y_test_binary, y_score)

    return auc_score