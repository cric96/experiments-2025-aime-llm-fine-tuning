import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, GEval
from tests.test_rag_metrics import actual_output

from adapter import CustomGeminiFlash
from core import GeminiService

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load CLI with API key, data, train/val division, and augmentation amount")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the service")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file")
    parser.add_argument("--sampling_ratio", type=float, default=0.5, help="Proportion of the data to use for sampling")
    parser.add_argument("--model", type=str, default="llama1b", help="Model to use for embedding")

    args = parser.parse_args()
    df = pd.read_csv(args.data_path + "/" + args.model + "/" + "validation_predictions.csv")
    ground_truth = df["Response"].tolist()
    generated = df["pred_Response"].tolist()
    sentences = df["Sentence"].tolist()
    # generate embeddings for the ground truth and generated responses
    client = GeminiService(args.api_key)
    embeddings_ground_truth = client.embed_contents("models/text-embedding-004", ground_truth)
    embeddings_generated = client.embed_contents("models/text-embedding-004", generated)
    # Compute the cosine similarity
    similarity_scores = []
    for i in range(len(embeddings_ground_truth)):
        similarity_scores.append(np.dot(embeddings_ground_truth[i], embeddings_generated[i]) / (np.linalg.norm(embeddings_ground_truth[i]) * np.linalg.norm(embeddings_generated[i])))

    # Compute the mean similarity score
    mean_similarity_score = np.mean(similarity_scores)
    print(f"Mean similarity score: {mean_similarity_score}")

    pca = PCA(n_components=3)
    pca.fit(embeddings_ground_truth + embeddings_generated)
    data_ground_truth = pca.transform(embeddings_ground_truth)
    data_generated = pca.transform(embeddings_generated)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data_ground_truth[:, 0], data_ground_truth[:, 1], data_ground_truth[:, 2], c="blue")
    ax.scatter(data_generated[:, 0], data_generated[:, 1], data_generated[:, 2], c="red", alpha=0.4)
    plt.show()

    metrics = [
        #AnswerRelevancyMetric(
        #    threshold=0.7,
        #    model=CustomGeminiFlash(args.api_key),
        #    include_reason=True
        #),
        GEval(
            name="Correctness",
            criteria="Determine whether the actual output is factually correct based on the expected output.",
            model=CustomGeminiFlash(args.api_key),
            # NOTE: you can only provide either criteria or evaluation_steps, and not both
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail",
                "Vague language, or contradicting OPINIONS, are OK"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        )
    ]

    test_cases = []
    for (sentence, ground_truth, generated) in zip(sentences, ground_truth, generated):
        test_cases.append(
            LLMTestCase(
                input=sentence,
                actual_output=ground_truth,
                expected_output=generated
            )
        )
    results = evaluate(test_cases, metrics, ignore_errors=True, skip_on_missing_params=True, print_results=False)

    metrics_computed = [metrics for metrics in results.test_results]
    all_data = [metric.metrics_data[0].score for metric in metrics_computed]
    sum = 0.0
    i = 0
    all_data_normalized = []
    for data in all_data:
        if data is not None:
            sum += data
            all_data_normalized.append(data)
        else:
            all_data_normalized.append(0.0)

        i += 1
    print(f"Mean correctness score: {sum / i}")

    # store the results
    results.to_csv(args.output_path)
