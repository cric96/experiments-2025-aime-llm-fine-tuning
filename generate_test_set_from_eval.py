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

if __name__ == "__main__":
    # load all_models
    all_models = json.load(open("generated/all_models.json"))
    a = all_models["llama1b-fine-tuned"]
    sampling_ratio = 0.3
    all_data = []
    for model in all_models:
        print(f"Processing model {model}")
        all_indicies = np.arange(len(all_models[model]["sentences"]))
        # sample the data
        np.random.shuffle(all_indicies)
        sampled_indicies = all_indicies[:int(len(all_indicies) * sampling_ratio)]
        print(sampled_indicies)
        # get the sentences
        sentences = [all_models[model]["sentences"][i] for i in sampled_indicies]
        # get the responses
        responses = [all_models[model]["generated"][i] for i in sampled_indicies]
        print(f"Number of sentences {len(sentences)}")
        print(f"Number of responses {len(responses)}")
        # create the dataframe
        models = [model] * len(sentences)
        print(f"Number of models {len(models)}")
        df = pd.DataFrame({"Sentence": sentences, "Response": responses, "Model": models})
        all_data.append(df)
    # create the dataframe

    df = pd.concat(all_data)
    # shuffle
    df = df.sample(frac=1)
    # save the data
    df.to_csv("generated/randomized_test.csv", index=False)