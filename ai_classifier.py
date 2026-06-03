

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
    Reads the CSV and keeps tweet_id, tweet_text, and class.
    The original class is renamed to real_class.
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
    Sends one tweet to Ollama using the provided model and returns the predicted class.
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
    Sends one tweet to Ollama using the provided model and returns the predicted class.
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
    Builds confusion matrix metrics and calculates overall global metrics.
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
    Classifies every tweet using the provided model and adds the Ollama prediction
    as a new column called new_class.
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
    Saves the results to a CSV file.
    Output columns:
    tweet_id, tweet_text, real_class, new_class
    """
    output_path = resolve_output_path(output_file)

    output_columns = ["tweet_id", "tweet_text", "real_class", "new_class"]

    df[output_columns].to_csv(output_path, index=False, encoding="utf-8")

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    input_file = "data_train(in).csv"
    output_file = "llama_32_predictions.csv"

    tweets_df = read_tweets(input_file)

    results_df = classify_tweets(tweets_df, model="llama3.2:latest", few_shot=True)

    print(results_df.head())
    save_results(results_df, output_file)
    
    cm_df = calculate_confusion_matrix(output_file)