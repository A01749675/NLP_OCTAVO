

"""Módulo para clasificar tweets usando un modelo de Ollama.

Este módulo lee un CSV con tweets y sus etiquetas reales, envía cada tweet a
un modelo de Ollama para obtener una predicción de clase, y contiene utilidades
para evaluar el rendimiento de la clasificación.
"""

import ollama
import pandas as pd
from paths import resolve_input_path, resolve_output_path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def read_tweets(file_path):
    """
    Reads a CSV file containing tweets and keeps specific columns.

    This function resolves the given file path, reads the CSV data, and
    extracts the 'tweet_id', 'tweet_text', and 'class' columns. It drops
    any rows with missing values in these columns and renames the 'class'
    column to 'real_class'.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to be read.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the cleaned data with columns 'tweet_id',
        'tweet_text', and 'real_class'.

    Raises
    ------
    ValueError
        If any of the required columns ('tweet_id', 'tweet_text', 'class')
        are missing from the input CSV file.
    """
    file_path = resolve_input_path(file_path)

    df = pd.read_csv(file_path, encoding="utf-8")

    required_columns = ["tweet_id", "tweet_text", "class"]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in CSV: {missing_columns}")

    df = df[required_columns].dropna()

    df = df.rename(columns={
        "class": "real_class"
    })

    return df


def classify_tweet(tweet_text, model="gemma2:9b"):
    """
    Sends a single tweet to an Ollama model for classification and returns the predicted class.

    This function prompts an LLM via the Ollama API to classify a given tweet
    into one of two mutually exclusive categories: 'control' or 'anorexia'.
    It parses the model's response, strictly extracting the class name, and
    defaults to 'control' if the output is ambiguous or unidentifiable.

    Parameters
    ----------
    tweet_text : str
        The text content of the tweet to be classified.
    model : str, optional
        The name of the Ollama model to use for classification (default is "gemma2:9b").

    Returns
    -------
    str
        The predicted class label, which will be either "anorexia" or "control".
    """
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"""
Classify the following tweet into exactly one of these classes:

control
anorexia

YOU CANNOT RETURN ANYTHING ELSE THAN THE CLASS NAME, DO NOT RETURN ANY EXPLANATION OR ANY OTHER TEXT. 
YOU CANNOT RETURN ANOTHER CLASS NAME, ONLY ONE OF THE TWO CLASSES.

Return only the class name. Do not explain.

Tweet:
{tweet_text}
""",
            }
        ],
    )

    prediction = response["message"]["content"].strip().lower()

    if "anorexia" in prediction:
        return "anorexia"
    elif "control" in prediction:
        return "control"
    else:
        return "control"


def classify_tweet_few_shot(tweet_text, model="gemma2:9b"):
    """
    Sends a single tweet to an Ollama model for classification using a few-shot prompt and returns the predicted class.

    This function prompts an LLM via the Ollama API to classify a given tweet
    into one of two mutually exclusive categories: 'control' or 'anorexia'.
    It utilizes a few-shot prompting approach by providing several labeled
    examples within the prompt to guide the model's behavior. It parses the
    model's response, strictly extracting the class name, and defaults to
    'control' if the output is ambiguous or unidentifiable.

    Parameters
    ----------
    tweet_text : str
        The text content of the tweet to be classified.
    model : str, optional
        The name of the Ollama model to use for classification (default is "gemma2:9b").

    Returns
    -------
    str
        The predicted class label, which will be either "anorexia" or "control".
    """
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"""
Classify the following tweet into exactly one of these classes:

control
anorexia

Here are some examples:
    Cheesecake saludable sin azúcar y sin lactosa  con mermerlada casera de moras,control
    Mañana empiezo otra vez!,anorexia
    12 hs sin comer #anoreccia  #bulymia #amarme,anorexia
    "Cuando la gente me pregunta, puedo tomar Coca-Cola light? Aquí tenéis la respuesta. Insisto siempre en que desde que existen los alimentos “light” la obesidad en el mundo no ha hecho más que crecer.  only",control
    que le pasa a tu cuerpo cuando no tomas agua |  Carla Diaz TV https://youtu.be/6ROyt9VH4Oo  #agua #tea #té #adelgazar #bebidas  #metabolismo #deshidratacion #hidratacion,control
    "Un revuelto y un pedazo de tomate aliñado.  Si esque para cuidarse solo hay que quererlo., control
    #runhappy",control
    "Nada mejor que correr 5km  y haser unas series de sentadillas para quitar la gripe y la temperatura.#ejercicio #calisthenics  #noche #SIGUEMEYTESIGO",control
    "Un buen desayuno y listo para un día intenso!, control 
    Una opción saludable y muy nutritiva para media tarde: tostada de #aguacate  #HealthyFood,control
    "Todavía piensas que comer sano es aburrido? Hoy para comer, una deliciosa berenjena a la parmesana!!  ing   #vidasana #habitosaludables #cadiz #sindietas #perderpeso",control
    "Yo soy mía, y estoy mejorando para no serlo más, y me pone mal q los demás crean esto de una forma de viví o sobrevivir, no lo es una puta mentira. Lo q están haciendo no es evitar engordar sino que están evitando vivír",anorexia
    "No se porque, pero me siento igual o peor a cuando empece en esto me veo igual de vaca #fat #thinspo #thinspiration #gorda #proana  #anorexic #bulimic",anorexia
    Solo Deseo estar #Delgada Solo Eso,anorexia

YOU CANNOT RETURN ANYTHING ELSE THAN THE CLASS NAME, DO NOT RETURN ANY EXPLANATION OR ANY OTHER TEXT. 
YOU CANNOT RETURN ANOTHER CLASS NAME, ONLY ONE OF THE TWO CLASSES.

Return only the class name. Do not explain.

Tweet:
{tweet_text}
""",
            }
        ],
    )

    prediction = response["message"]["content"].strip().lower()

    if "anorexia" in prediction:
        return "anorexia"
    elif "control" in prediction:
        return "control"
    else:
        return "control"


def classify_chain_of_thought(tweet_text, model="gemma2:9b"):
    """
    Classifies the tweet using a chain of thought prompting approach.
    """
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"""Classify the following tweet into exactly one of these classes:
control
anorexia
Here are some examples:
    Cheesecake saludable sin azúcar y sin lactosa  con mermerlada casera de moras,control
    Mañana empiezo otra vez!,anorexia
    
"""
            }
        ],
    )

    prediction = response["message"]["content"].strip().lower()

    if "anorexia" in prediction:
        return "anorexia"
    elif "control" in prediction:
        return "control"
    else:
        return "control"



def calculate_confusion_matrix(output_file):
    """
    Calculates, prints, and returns the confusion matrix and overall classification metrics.

    This function reads a CSV file containing true ('real_class') and
    predicted ('new_class') labels. It calculates various performance
    metrics (accuracy, macro/weighted precision, recall, and F1-score)
    assuming the target classes are 'control' and 'anorexia'. It prints
    a detailed evaluation report to the console and returns the confusion
    matrix as a formatted DataFrame.

    Parameters
    ----------
    output_file : str
        The path to the CSV file containing the classification results.

    Returns
    -------
    pandas.DataFrame
        A DataFrame representing the confusion matrix, with rows indicating
        true classes and columns indicating predicted classes.

    Raises
    ------
    ValueError
        If the required columns ('real_class', 'new_class') are missing
        from the input CSV file.
    """
    output_path = resolve_output_path(output_file)
    df = pd.read_csv(output_path, encoding="utf-8")

    required_columns = ["real_class", "new_class"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in output file: {missing_columns}")

    y_true = df["real_class"]
    y_pred = df["new_class"]
    labels = ["control", "anorexia"]

    metrics = calculate_confusion_metrics(y_true, y_pred, labels=labels)
    cm = metrics["confusion_matrix"].values
    metric_labels = metrics["labels"]

    cm_df = pd.DataFrame(
        cm,
        index=[f"real_{label}" for label in metric_labels],
        columns=[f"predicted_{label}" for label in metric_labels],
    )

    print("=========================================")
    print("      OVERALL CLASSIFICATION RESULTS      ")
    print("=========================================")
    print(f"Total Samples Evaluated: {len(y_true)}")
    print(f"Overall Accuracy:       {metrics['accuracy']:.4f}")

    overall = metrics["overall"]
    print("\n--- Macro Averages (Unweighted) ---")
    print(f"Macro Precision:        {overall['macro_precision']:.4f}")
    print(f"Macro Recall:           {overall['macro_recall']:.4f}")
    print(f"Macro F1-Score:         {overall['macro_f1']:.4f}")

    print("\n--- Weighted Averages (Weighted by Support) ---")
    print(f"Weighted Precision:     {overall['weighted_precision']:.4f}")
    print(f"Weighted Recall:        {overall['weighted_recall']:.4f}")
    print(f"Weighted F1-Score:      {overall['weighted_f1']:.4f}")

    print("\nConfusion Matrix:")
    print(cm_df)
    print("=========================================")

    return cm_df


def calculate_confusion_metrics(y_true, y_pred, labels=None):
    """
    Builds a confusion matrix and calculates detailed classification metrics.

    This function computes per-label metrics (precision, recall, F1-score,
    and support) as well as overall global metrics (accuracy, macro averages,
    and weighted averages) to evaluate the performance of a classification
    model. It dynamically handles missing labels and appends them to the
    evaluation if a specific subset is provided but other labels are present
    in the data.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated target values as returned by a classifier.
    labels : array-like, optional
        List of labels to index the matrix and calculate metrics for. If None,
        it defaults to the sorted union of unique labels present in `y_true`
        and `y_pred`.

    Returns
    -------
    dict
        A comprehensive dictionary containing the evaluation results:
        - 'confusion_matrix' (pandas.DataFrame): The confusion matrix.
        - 'accuracy' (float): The overall accuracy score.
        - 'per_label' (dict): Precision, recall, F1-score, and support for each label.
        - 'labels' (list): The complete list of labels evaluated.
        - 'overall' (dict): Macro and weighted averages for precision, recall, and F1-score.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    all_labels = list(labels)
    extra_labels = [l for l in sorted(set(y_true) | set(y_pred)) if l not in all_labels]
    all_labels.extend(extra_labels)

    # 1. Calculate Per-Label metrics
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    precision, recall, f1_score, support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, zero_division=0
    )

    per_label = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1_score": float(f1_score[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(all_labels)
    }

    # 2. CALCULATE OVERALL GLOBAL METRICS (Not divided by class)
    # Macro average treats all classes equally
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    # Weighted average accounts for class imbalance (support)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    matrix = pd.DataFrame(
        cm,
        index=[f"real_{label}" for label in all_labels],
        columns=[f"predicted_{label}" for label in all_labels],
    )

    return {
        "confusion_matrix": matrix,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "per_label": per_label,
        "labels": all_labels,
        "overall": {
            "macro_precision": float(macro_p),
            "macro_recall": float(macro_r),
            "macro_f1": float(macro_f1),
            "weighted_precision": float(weighted_p),
            "weighted_recall": float(weighted_r),
            "weighted_f1": float(weighted_f1),
        }
    }


def classify_tweets(df, model="gemma2:9b",few_shot=False):
    """
    Classifies a collection of tweets in a DataFrame and appends the predictions.

    This function iterates over each row in the input DataFrame, extracts
    the tweet text, and passes it to the `classify_tweet` function using
    the specified model. It prints the classification progress to the console
    and returns a copy of the DataFrame with an additional 'new_class' column
    containing the model's predictions.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the tweets to be classified. It is
        expected to contain at least the columns 'tweet_id', 'tweet_text',
        and 'real_class'.
    model : str, optional
        The name of the Ollama model to use for classification (default is "gemma2:9b").

    Returns
    -------
    pandas.DataFrame
        A copy of the original DataFrame with a new column named 'new_class'
        that contains the predicted class labels for each tweet.
    """
    new_classes = []

    for _, row in df.iterrows():
        tweet_id = row["tweet_id"]
        tweet_text = row["tweet_text"]
        real_class = row["real_class"]

        print(f"Classifying tweet_id: {tweet_id} | real class: {real_class}")

        # print(f"Tweet text: {tweet_text}")
        if few_shot:
            new_class = classify_tweet_few_shot(tweet_text, model=model)
        else:
            new_class = classify_tweet(tweet_text, model=model)
        new_classes.append(new_class)
        # print(f"Predicted class: {new_class}")

    result_df = df.copy()
    result_df["new_class"] = new_classes

    return result_df


def save_results(df, output_file):
    """
    Saves the classification results from a DataFrame to a CSV file.

    This function resolves the designated output path, filters the DataFrame
    to keep only the essential columns ('tweet_id', 'tweet_text', 'real_class',
    and 'new_class'), and exports the data to a comma-separated values (CSV)
    file. It also prints a confirmation message to the console with the final
    save path.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the classified tweets. It must include the
        columns 'tweet_id', 'tweet_text', 'real_class', and 'new_class'.
    output_file : str
        The destination file name or path where the CSV will be saved.

    Returns
    -------
    None
    """
    output_path = resolve_output_path(output_file)

    output_columns = ["tweet_id", "tweet_text", "real_class", "new_class"]

    df[output_columns].to_csv(output_path, index=False, encoding="utf-8")

    print(f"Results saved to: {output_path}")


def main():
    """
    Executes the full pipeline for tweet classification and evaluation.

    This function serves as the main entry point for the script. It defines
    the input and output file paths, reads the training data, performs
    classification using a specified Ollama model (in this case,
    "llama3.2:latest"), prints a preview of the results, saves the
    predictions to a CSV file, and calculates the final confusion matrix.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    input_file = "data_train(in).csv"
    output_file = "llama_32_predictions_few.csv"

    tweets_df = read_tweets(input_file)

    results_df = classify_tweets(tweets_df, model="llama3.2:latest", few_shot=True)

    print(results_df.head())
    save_results(results_df, output_file)

    cm_df = calculate_confusion_matrix(output_file)


if __name__ == "__main__":
    main()