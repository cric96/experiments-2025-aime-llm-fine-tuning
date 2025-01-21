import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.cluster import MeanShift
from sklearn.decomposition import PCA

from core import GeminiService

def augment(api_key, data_path, sampling_ratio, output_path):
    df = load_data(data_path)
    gemini_service = GeminiService(api_key)
    embeddings = embed_data(gemini_service, df)
    pca_transformed = perform_pca(embeddings)
    mean_shift = perform_clustering(pca_transformed)
    sampled_sentences, sampled_indices = sample_data(df, mean_shift, sampling_ratio)
    knowledge = " ".join(sampled_sentences)
    all_prompts = generate_prompts(knowledge, sampled_sentences)
    response = generate_responses(gemini_service, all_prompts)
    cleaned_responses = clean_responses(response)
    flatten_sentences, flatten_responses = flatten_responses_data(cleaned_responses, df, sampled_indices)
    df_generated, df_not_sampled = create_dataframes(flatten_sentences, flatten_responses, df, sampled_indices)
    store_dataframes(df_generated, df_not_sampled, output_path)

def load_data(data_path):
    df = pd.read_csv(data_path)
    df["text"] = "### Sentence ### " + df["Sentence"] + " ### Response ### " + df["Response"]
    return df

def embed_data(gemini_service, df):
    return gemini_service.embed_contents("models/text-embedding-004", df["text"].tolist())

def perform_pca(embeddings, n_components=3):
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    return pca.transform(embeddings)

def perform_clustering(data):
    mean_shift = MeanShift()
    mean_shift.fit(data)
    return mean_shift

def sample_data(df, mean_shift, sampling_ratio):
    sampled_sentences = []
    sampled_indices = []
    for cluster in range(max(mean_shift.labels_) + 1):
        cluster_indices = df.index[mean_shift.labels_ == cluster]
        n_samples = int(len(cluster_indices) * sampling_ratio)
        random_sample = np.random.choice(cluster_indices, n_samples, replace=False)
        sampled_indices.extend(random_sample)
        sampled_sentences.extend(df["text"][random_sample])
    return sampled_sentences, sampled_indices

def generate_prompts(knowledge, sampled_sentences):
    return [prompt_generation(knowledge, text) for text in sampled_sentences]

def generate_responses(gemini_service, all_prompts, max_retries=3, sleep_time=3):
    response = []
    current_prompt = 0
    for prompt in all_prompts:
        for attempt in range(max_retries):
            print(f"Generating content for prompt: {current_prompt}")
            try:
                response.append(gemini_service.generate_content(
                    model='gemini-2.0-flash-exp',
                    content=prompt
                ))
                break  # Exit the retry loop if the request is successful
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(sleep_time)  # Wait before retrying
                else:
                    print(f"Failed to generate content for prompt after {max_retries} attempts: {e}")
        current_prompt += 1
    return response

def clean_responses(response):
    cleaned_responses = []
    for r in response:
        response_text = r.replace("```json", "").replace("```", "")
        response_json = json.loads(response_text)
        cleaned_responses.append(response_json)
    return cleaned_responses

def flatten_responses_data(cleaned_responses, df, sampled_indices):
    flatten_sentences = []
    flatten_responses = []
    for r in cleaned_responses:
        for data in r:
            flatten_sentences.append(data["sentence"])
            flatten_responses.append(data["response"])
    flatten_sentences.extend(df["Sentence"][sampled_indices])
    flatten_responses.extend(df["Response"][sampled_indices])
    return flatten_sentences, flatten_responses

def create_dataframes(flatten_sentences, flatten_responses, df, sampled_indices):
    df_generated = pd.DataFrame({
        "Sentence": flatten_sentences,
        "Response": flatten_responses,
    })
    df_not_sampled = df.drop(sampled_indices)
    return df_generated, df_not_sampled

def store_dataframes(df_generated, df_not_sampled, output_path):
    df_generated.to_csv(output_path + "/generated.csv", index=False)
    df_not_sampled.to_csv(output_path + "/test.csv", index=False)

def prompt_generation(knowledge, text):
    return f"""
    I will share with you a dataset of responses to sentences. In this form:
    ### Sentence ### <knowledge> ### Response ### <text>
    After this knowledge, I will share just one of those sentences with you.
    Your task it to generate five different sentence - response pairs that could follow the sentence I shared with you but that should be different from the responses in the dataset.
    You will reply with a json object with the following structure:

    Use this JSON schema for the reply:

    Data = {{'sentence': str, 'response': str}}
    Return: list[Data]

    knowledge:
    {knowledge}
    This is the pair used to generate five different sentence - response pairs (use the same tone and style as the dataset):
    {text}
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load CLI with API key, data, train/val division, and augmentation amount")
    parser.add_argument("--api_key", type=str, required=True, help="API key for the service")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file")
    parser.add_argument("--sampling_ratio", type=float, default=0.5, help="Proportion of the data to use for sampling")

    args = parser.parse_args()
    augment(args.api_key, args.data_path, args.sampling_ratio, args.output_path)