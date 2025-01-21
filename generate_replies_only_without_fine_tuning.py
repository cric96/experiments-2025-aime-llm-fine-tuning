import argparse

import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, GEval
from tests.test_rag_metrics import actual_output

from adapter import CustomGeminiFlash
from core import GeminiService, OllamaService

def choose_model(model, api_key):
    if model in "gemini":
        return GeminiService(api_key)
    else:
        return OllamaService()

def parse_arguments():
    parser = argparse.ArgumentParser(description="It is used to generate replies without fine-tuning, change only the model and pass the api key")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the service")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file")
    parser.add_argument("--reference", type=str, default="llama1b-fine-tuned", help="Base dataset to use for the validation")
    parser.add_argument("--model", type=str, default="llama3b-prompt", help="Model for creating the responses")
    return parser.parse_args()

def load_data(data_path, reference):
    df = pd.read_csv(data_path + "/" + reference + "/" + "validation_predictions.csv")
    ground_truth = df["Response"].tolist()
    generated = df["pred_Response"].tolist()
    sentences = df["Sentence"].tolist()
    return df, ground_truth, generated, sentences

def generate_responses(service, system_prompt, generated):
    return [service.generate_content(args.model, system_prompt + sentence) for sentence in generated]

def save_data(df, output_path, model):
    if not os.path.exists(output_path + "/" + model):
        os.makedirs(output_path + "/" + model)
    df.to_csv(output_path + "/" + model + "/validation_predictions.csv", index=False)

if __name__ == "__main__":
    args = parse_arguments()
    system_prompt = "You are a chatbot to support hypertensive patients, reply concisely to the following question:"
    df, ground_truth, generated, sentences = load_data(args.data_path, args.reference)
    service = choose_model(args.model, args.api_key)
    service_generated = generate_responses(service, system_prompt, generated)
    df["pred_Response"] = service_generated
    save_data(df, args.output_path, args.model)
