import argparse

import numpy as np
import os
import pandas as pd
import seaborn as sns
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from matplotlib import pyplot as plt, cm
from sklearn.decomposition import PCA
import json
from adapter import CustomGeminiFlash
from core import GeminiService


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate the correctness of the generated responses")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the service")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file")
    parser.add_argument("--cache", type=str, help="Cache used to compute all the relevant information")
    return parser.parse_args()

def load_data(data_path, model):
    df = pd.read_csv(f"{data_path}/{model}/validation_predictions.csv")
    ground_truth = df["Response"].tolist()
    generated = df["pred_Response"].tolist()
    sentences = df["Sentence"].tolist()
    return df, ground_truth, generated, sentences

def generate_embeddings(client, ground_truth, generated):
    embeddings_ground_truth = client.embed_contents("models/text-embedding-004", ground_truth)
    embeddings_generated = client.embed_contents("models/text-embedding-004", generated)
    return embeddings_ground_truth, embeddings_generated

def compute_similarity(embeddings_ground_truth, embeddings_generated):
    similarity_scores = [
        np.dot(embeddings_ground_truth[i], embeddings_generated[i]) /
        (np.linalg.norm(embeddings_ground_truth[i]) * np.linalg.norm(embeddings_generated[i]))
        for i in range(len(embeddings_ground_truth))
    ]
    return similarity_scores, np.mean(similarity_scores)

def plot_embeddings(embeddings_ground_truth, embeddings_generated, output_path, model):
    pca = PCA(n_components=3)
    pca.fit(embeddings_ground_truth)
    data_ground_truth = pca.transform(embeddings_ground_truth)
    data_generated = pca.transform(embeddings_generated)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_ground_truth[:, 0], data_ground_truth[:, 1], data_ground_truth[:, 2], c="blue")
    ax.scatter(data_generated[:, 0], data_generated[:, 1], data_generated[:, 2], c="red", alpha=0.4)
    # store
    plt.savefig(f"{output_path}/embeddings-{model}.png")
    # plot density in 3D
    plt.close()
    df_pca_ground_truth = pd.DataFrame(data_ground_truth, columns=["x", "y", "z"])
    df_pca_generated = pd.DataFrame(data_generated, columns=["x", "y", "z"])

    color_from_cmap = cm.get_cmap("viridis")
    sns.kdeplot(df_pca_ground_truth,
                shade=False,
                x="x",
                alpha=0.5,
                label="Ground truth",
                )
    sns.kdeplot(df_pca_generated,
                shade=True,
                x="x",
                alpha=0.5,
                label="Generated")
    plt.savefig(f"{output_path}/density-{model}.png")

def evaluate_metrics(api_key, sentences, ground_truth, generated):
    metrics = [
        GEval(
            name="Correctness",
            criteria="Determine whether the actual output is factually correct based on the expected output.",
            model=CustomGeminiFlash(api_key),
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail",
                "Vague language, or contradicting OPINIONS, are OK"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        )
    ]
    test_cases = [
        LLMTestCase(input=sentence, actual_output=generated, expected_output=ground_truth)
        for sentence, ground_truth, generated in zip(sentences, ground_truth, generated)
    ]
    results = evaluate(test_cases, metrics, ignore_errors=True, skip_on_missing_params=True, print_results=False)
    return results

def compute_correctness_statistics(results):
    metrics_computed = [metrics for metrics in results.test_results]
    all_data = [metric.metrics_data[0].score for metric in metrics_computed]
    sum_scores = sum(data for data in all_data if data is not None)
    count = len(all_data)
    return all_data, sum_scores / count if count else 0

def main():
    args = parse_arguments()
    # take all model from data (listing the folder)
    all_models = os.listdir(args.data_path)
    # take only folder
    all_models = [model for model in all_models if os.path.isdir(f"{args.data_path}/{model}")]
    all_models_loaded = {}
    if args.cache is not None:
        print("Cache loaded")
        with open(args.cache, "r") as f:
            all_models_loaded = json.load(f)
    else:
        print("Cache not loaded, computing all the data")
        print(all_models)
        all_models_loaded = {}
        for model in all_models:
            df, ground_truth, generated, sentences = load_data(args.data_path, model)
            client = GeminiService(args.api_key)
            embeddings_ground_truth, embeddings_generated = generate_embeddings(client, ground_truth, generated)
            similarities, mean = compute_similarity(embeddings_ground_truth, embeddings_generated)
            results = evaluate_metrics(args.api_key, sentences, ground_truth, generated)
            all_score, mean_correctness_score = compute_correctness_statistics(results)

            all_models_loaded[model] = {
                "embeddings_ground_truth": embeddings_ground_truth,
                "embeddings_generated": embeddings_generated,
                "similarities": similarities,
                "mean_similarity": mean,
                "sentences": sentences,
                "ground_truth": ground_truth,
                "generated": generated,
                "all_score": all_score,
            }
            with open(f"{args.output_path}/results-{model}.json", "w") as f:
                data = results.model_dump_json()
                json.dump(data, f)
            # plot embedding and density
            plot_embeddings(embeddings_ground_truth, embeddings_generated, args.output_path, model)
            # save the data
            print(f"Mean correctness score for {model}: {mean_correctness_score}")
            # save the data
        with open(f"{args.output_path}/all_models.json", "w") as f:
            json.dump(all_models_loaded, f)

    # plot violin of similarity
    df_similarity = pd.DataFrame(all_models_loaded)
    # invert rows and columns
    df_similarity = df_similarity.transpose()
    print(df_similarity["similarities"])
    # convert to dict, with name and similarity
    to_plot = {}
    print(df_similarity)
    models = [
        "llama1b-fine-tuned",
        "llama1b-prompt",
        "gemma-2b-fine-tuned",
        "gemma-2b-prompt",
        "llama3b-fine-tuned",
        "llama3b-prompt",
        "qwen-3b-fine-tuned",
        "qwen-3b-prompt",
        "phi-3b-fine-tuned",
        "phi-3b-prompt",
        "gemini-1.5-pro"
    ]
    models_kde = [
        "llama1b-fine-tuned",
        "llama3b-fine-tuned",
        "gemma-2b-fine-tuned",
        "qwen-3b-fine-tuned",
        "phi-3b-fine-tuned",
        "gemini-1.5-pro",
        "llama1b-prompt",
        "llama3b-prompt",
        "gemma-2b-prompt",
        "qwen-3b-prompt",
        "phi-3b-prompt",
    ]
    label_mapping = {
        "llama1b-fine-tuned": "llama-1b (T)",
        "llama1b-prompt": "llama-1b (P)",
        "gemma-2b-fine-tuned": "gemma-2b (T)",
        "gemma-2b-prompt": "gemma-2b (P)",
        "llama3b-fine-tuned": "llama-3b (T)",
        "llama3b-prompt": "llama-3b (P)",
        "qwen-3b-fine-tuned": "qwen-3b (T)",
        "qwen-3b-prompt": "qwen-3b (P)",
        "phi-3b-fine-tuned": "phi-3b (T)",
        "phi-3b-prompt": "phi-3b (P)",
        "gemini-1.5-flash": "gemini-1.5 (flash)",
        "gemini-1.5-pro": "gemini-1.5 (pro)",
    }

    for model in models:
        to_plot[label_mapping[model]] = df_similarity["similarities"][model]
    sns.violinplot(data=to_plot, split=True, inner="quart")
    # increase font size for xlabel
    plt.xticks(rotation=45, fontsize=14)

    plt.tight_layout()
    #plt.show()
    plt.savefig(f"{args.output_path}/violin-similarity.pdf")
    plt.close()
    to_plot = {}
    for model in models:
        to_plot[label_mapping[model]] = df_similarity["all_score"][model]
    # plot violin of correctness
    sns.violinplot(data=to_plot, split=True, inner="quart")
    plt.xticks(rotation=45, fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{args.output_path}/violin-correctness.pdf")
    plt.close()
    all_embeddings = df_similarity["embeddings_ground_truth"]["llama1b-fine-tuned"]

    # Apply PCA to the combined data
    pca = PCA(n_components=2)
    pca.fit(all_embeddings)

    # Prepare data for plotting
    plot_data = []

    for model in ["reference"] + models_kde:
        embeddings = (
            df_similarity["embeddings_ground_truth"]["llama1b-fine-tuned"]
            if model == "reference"
            else df_similarity["embeddings_generated"][model]
        )
        transformed_embeddings = pca.transform(embeddings)
        model_df = pd.DataFrame(transformed_embeddings, columns=["PC1", "PC2"])
        model_df["K"] = label_mapping.get(model, model)
        model_df["Group"] = "Reference" if model == "reference" else "Generated"
        model_df["Group"] = "Tuned" if "tuned" in model else model_df["Group"]
        model_df["Group"] = "Prompt" if "prompt" in model else model_df["Group"]
        model_df["Group"] = "Prompt" if "gemini" in model else model_df["Group"]

        plot_data.append(model_df)

    # Combine data for plotting
    plot_df = pd.concat(plot_data, ignore_index=True)
    # Combine all into a single DataFrame
    g = sns.FacetGrid(plot_df, hue="Group", col="K", col_wrap=6, dropna=True, sharex=True, sharey=True, height=1.5)
    g.map(sns.kdeplot, "PC1", "PC2", shade=True)
    g.set(xlim=(-1, 1), ylim=(-1, 1))
    # remove xaxis labels
    for ax in g.axes.flatten():
        ax.set_xlabel("")
        ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(f"{args.output_path}/kdeplot-grid.pdf")
    #plt.show()

if __name__ == "__main__":
    main()