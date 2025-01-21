import argparse

import numpy as np
import pandas as pd
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import json
from adapter import CustomGeminiFlash
from core import GeminiService


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate the correctness of the generated responses")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the service")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file")
    parser.add_argument("--sampling_ratio", type=float, default=0.5, help="Proportion of the data to use for sampling")
    parser.add_argument("--model", type=str, default="llama3b-prompt", help="Model to use for embedding")
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
    return np.mean(similarity_scores)

def plot_embeddings(embeddings_ground_truth, embeddings_generated, output_path, model):
    pca = PCA(n_components=3)
    pca.fit(embeddings_ground_truth + embeddings_generated)
    data_ground_truth = pca.transform(embeddings_ground_truth)
    data_generated = pca.transform(embeddings_generated)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_ground_truth[:, 0], data_ground_truth[:, 1], data_ground_truth[:, 2], c="blue")
    ax.scatter(data_generated[:, 0], data_generated[:, 1], data_generated[:, 2], c="red", alpha=0.4)
    # store
    plt.savefig(f"{output_path}/embeddings-{model}.png")

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
        LLMTestCase(input=sentence, actual_output=ground_truth, expected_output=generated)
        for sentence, ground_truth, generated in zip(sentences, ground_truth, generated)
    ]
    results = evaluate(test_cases, metrics, ignore_errors=True, skip_on_missing_params=True, print_results=False)
    return results

def compute_mean_correctness(results):
    metrics_computed = [metrics for metrics in results.test_results]
    all_data = [metric.metrics_data[0].score for metric in metrics_computed]
    sum_scores = sum(data for data in all_data if data is not None)
    count = len(all_data)
    return sum_scores / count if count else 0

def main():
    args = parse_arguments()
    df, ground_truth, generated, sentences = load_data(args.data_path, args.model)
    client = GeminiService(args.api_key)
    embeddings_ground_truth, embeddings_generated = generate_embeddings(client, ground_truth, generated)
    mean_similarity_score = compute_similarity(embeddings_ground_truth, embeddings_generated)
    print(f"Mean similarity score: {mean_similarity_score}")
    plot_embeddings(embeddings_ground_truth, embeddings_generated, args.output_path, args.model)
    results = evaluate_metrics(args.api_key, sentences, ground_truth, generated)
    mean_correctness_score = compute_mean_correctness(results)
    print(f"Mean correctness score: {mean_correctness_score}")
    # convert result to json
    with open(f"{args.output_path}/results-{args.model}.json", "w") as f:
        data = results.model_dump_json()
        json.dump(data, f)
if __name__ == "__main__":
    main()