from sklearn.model_selection import train_test_split

from vectorizers import process_csv
from data_loader import get_data

from models.random_forest_model import get_model as get_random_forest
from models.logistic_regression_model import get_model as get_logistic_regression
from models.knn_model import get_model as get_knn
import pandas as pd 

from evaluation import (
    evaluate_model,
    print_metrics,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_random_forest_feature_importance,
    plot_logistic_regression_coefficients,
    plot_train_vs_test_accuracy_rf
)

PLOTS_ACTIVE = False

def select_model(model_name, random_state=42):
    """
    Selects the machine learning model according to the model name.

    Available models:
    - "rf" or "random_forest"
    - "lr" or "logistic_regression"
    - "knn"

    Parameters
    ----------
    model_name : str
        Name of the model to train.

    random_state : int
        Seed used for reproducibility.

    Returns
    -------
    model
        A scikit-learn compatible model.
    """

    model_name = model_name.lower()

    if model_name in ["rf", "random_forest", "random forest"]:
        return get_random_forest(random_state=random_state)

    elif model_name in ["lr", "logistic_regression", "logistic regression"]:
        return get_logistic_regression(random_state=random_state)

    elif model_name in ["knn", "k_nearest_neighbors", "k nearest neighbors"]:
        return get_knn()

    else:
        raise ValueError(
            "Invalid model_name. Use 'rf', 'lr', or 'knn'."
        )


def get_test_size(model_name):
    """
    Defines the train/test split according to the experimental design.

    Random Forest uses 70/30 to analyze overfitting.
    Logistic Regression and KNN use the general 80/20 convention.

    Parameters
    ----------
    model_name : str
        Name of the selected model.

    Returns
    -------
    float
        Test size value for train_test_split.
    """

    model_name = model_name.lower()

    if model_name in ["rf", "random_forest", "random forest"]:
        return 0.30

    return 0.20


def test_knn_model(input_file="data_train_cleaned.csv", random_state=42):

    model_name = "knn"
    vectorization_targets = ["tfidf", "ngrams","word2vec", "all"]
    performance = []
    for target in vectorization_targets:
        vectorized_file = process_csv(
            input_file=input_file,
            target=target
        )
    
        X, y = get_data(vectorized_file)

        # -----------------------------
        # 3. Split dataset
        # -----------------------------
        test_size = get_test_size(model_name)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        neighbors_list = [1, 2, 3, 5, 7, 9, 11, 15, 21]
        for n in neighbors_list:
            model = get_knn(n_neighbors=n)
            model.fit(X_train, y_train)
            y_pred, results = evaluate_model(
                model=model,
                X_test=X_test,
                y_test=y_test
            )
            print(f"KNN with {n} neighbors and {target} representation:")
            print_metrics(results)
            print("-" * 40)
            performance.append({
                "representation": target,
                "n_neighbors": n,
                **results
            })
    df = pd.DataFrame(performance)
    df.to_csv('knn_performance.csv', index=False)
    return performance


def train_and_plot(
    input_file="data_train_cleaned.csv",
    target="tfidf",
    model_name="rf",
    random_state=42
):
    """
    Runs the full training pipeline.

    Steps:
    1. Vectorizes the cleaned dataset.
    2. Loads X and y from the generated vectorized file.
    3. Splits the dataset.
    4. Selects the requested model.
    5. Trains the model.
    6. Evaluates the model.
    7. Generates plots according to the selected model.

    Parameters
    ----------
    input_file : str
        CSV file containing the cleaned tweets.

    target : str
        Vectorization target. Available values:
        - "tfidf"
        - "ngrams"
        - "all"

    model_name : str
        Model to train. Available values:
        - "rf"
        - "lr"
        - "knn"

    random_state : int
        Seed for reproducibility.

    Returns
    -------
    model
        Trained model.

    results : dict
        Dictionary containing evaluation metrics.
    """

    # -----------------------------
    # 1. Vectorization
    # -----------------------------
    vectorized_file = process_csv(
        input_file=input_file,
        target=target
    )

    if not vectorized_file:
        raise ValueError(
            "process_csv did not return a valid vectorized file name."
        )

    # -----------------------------
    # 2. Load dataset
    # -----------------------------
    X, y = get_data(vectorized_file)

    # -----------------------------
    # 3. Split dataset
    # -----------------------------
    test_size = get_test_size(model_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # -----------------------------
    # 4. Select model
    # -----------------------------
    model = select_model(
        model_name=model_name,
        random_state=random_state
    )

    # -----------------------------
    # 5. Train model
    # -----------------------------
    model.fit(X_train, y_train)

    # -----------------------------
    # 6. Evaluate model
    # -----------------------------
    y_pred, results = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test
    )

    print("\nExperiment Configuration")
    print("-" * 40)
    print(f"Input file: {input_file}")
    print(f"Vectorized file: {vectorized_file}")
    print(f"Representation: {target}")
    print(f"Model: {model_name}")
    print(f"Random state: {random_state}")
    print(f"Test size: {test_size}")
    print(f"Train size: {1 - test_size}")
    print("-" * 40)

    print_metrics(results)

    # -----------------------------
    # 7. General plots
    # -----------------------------
    
    if PLOTS_ACTIVE:
        plot_class_distribution(y)

        plot_confusion_matrix(
            y_test=y_test,
            y_pred=y_pred,
            title=f"Confusion Matrix - {model_name.upper()} with {target.upper()}"
        )

        # -----------------------------
        # 8. Model-specific plots
        # -----------------------------
        normalized_model_name = model_name.lower()

        if normalized_model_name in ["rf", "random_forest", "random forest"]:
            plot_random_forest_feature_importance(
                model=model,
                X=X,
                top_n=15
            )

            plot_train_vs_test_accuracy_rf(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                random_state=random_state
            )

        elif normalized_model_name in ["lr", "logistic_regression", "logistic regression"]:
            plot_logistic_regression_coefficients(
                model=model,
                X=X,
                top_n=15
            )

        elif normalized_model_name in ["knn", "k_nearest_neighbors", "k nearest neighbors"]:
            print(
                "KNN does not provide direct feature importances or coefficients."
            )

    return model, results


def run_experiments():
    """
    Runs multiple experiments using the combinations defined in the methodology.

    You can modify this list to test more combinations.
    """

    experiments = [
        {
            "target": "tfidf",
            "model_name": "rf"
        },
        {
            "target": "tfidf",
            "model_name": "lr"
        },
        {
            "target": "tfidf",
            "model_name": "knn"
        },
        {
            "target": "ngrams",
            "model_name": "rf"
        },
        {
            "target": "ngrams",
            "model_name": "lr"
        },
        {
            "target": "ngrams",
            "model_name": "knn"
        },
        {
            "target": "all",
            "model_name": "rf"
        },
        {
            "target": "all",
            "model_name": "lr"
        },
        {
            "target": "all",
            "model_name": "knn"
        },
        {
            "target": "word2vec",
            "model_name": "rf"
        },
        {
            "target": "word2vec",
            "model_name": "lr"
        },
        {
            "target": "word2vec",
            "model_name": "knn"
        }
    ]

    all_results = []

    for experiment in experiments:
        print("\n" + "=" * 60)
        print(
            f"Running experiment: {experiment['target']} + {experiment['model_name']}"
        )
        print("=" * 60)

        model, results = train_and_plot(
            input_file="data_train_cleaned.csv",
            target=experiment["target"],
            model_name=experiment["model_name"],
            random_state=42
        )

        results_record = {
            "representation": experiment["target"],
            "model": experiment["model_name"],
            **results
        }

        all_results.append(results_record)
    df = pd.DataFrame(all_results)
    df.to_csv('all_experiments.csv')
    
    return all_results


if __name__ == "__main__":
    # Run only one experiment
    # train_and_plot(
    #     input_file="data_train_cleaned.csv",
    #     target="all",
    #     model_name="rf",
    #     random_state=42
    # )

    # If you want to run all combinations, comment the previous block
    # and uncomment this:
    #
    results = run_experiments()
    print(results)
    
    # results_knn = test_knn_model()
    # print(results_knn)