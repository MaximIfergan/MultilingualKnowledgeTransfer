import pandas as pd
import random
import pandas as pd
# from mwviews.api import PageviewsClient
from jsonlines import jsonlines
from wikidata.client import Client
import pywikibot.data.api as api
import pywikibot
from tqdm.auto import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
import Data.DataPreprocessing as DataPreprocessing
import sys
import pickle
import re
import os
import EntityLinking.FinetuningDatasets.EntityStats as EntityStats
import Model.MLCBQA_Model as MLCBQA_Model

# ===============================      Global Variables:      ===============================

F1_SUCCESS = 0.5

# ===============================      Global functions:      ===============================

# ====================================      Class:      ====================================


class EmbeddingAnalysis:
    """ this class produce statistics analysis on the success of the model on the different languages """

    def __init__(self, embedding_layers, model_name, data_path, model_predictions):
        predictions = pd.read_csv(model_predictions)
        self.emb_layers = embedding_layers
        self.model_name = model_name
        self.results = pd.read_csv(data_path)
        self.results = self.results.loc[self.results['DataType'] == "dev"]
        self.results["Prediction"] = list(predictions["Generated Text"])
        self.results["F1"] = list(predictions["F1"])
        self.results["EM"] = list(predictions["EM"])

    def aggregate_dist_same_question_different_langs(self, dist_function):
        num_of_layers = 8
        encoder_dist = np.zeros(num_of_layers)
        decoder_dist = np.zeros(num_of_layers)
        example_num = 0
        random_encoder_dist = np.zeros(num_of_layers)
        random_decoder_dist = np.zeros(num_of_layers)
        random_example_num = 0
        df = self.results.loc[self.results['F1'] > F1_SUCCESS]  # only success answers
        ids = list(df["Id"].unique())
        for id in ids:
            langs = list(df.loc[df["Id"] == id]["Language"].unique())
            for i in range(len(langs)):
                lang_i_encoder_emb = self.emb_layers[id][langs[i]]["encoder_hidden_states"]
                lang_i_decoder_emb = self.emb_layers[id][langs[i]]["decoder_hidden_states"]
                for j in range(i + 1, len(langs)):
                    lang_j_encoder_emb = self.emb_layers[id][langs[j]]["encoder_hidden_states"]
                    lang_j_decoder_emb = self.emb_layers[id][langs[j]]["decoder_hidden_states"]
                    # TODO delete first and last tokens embedding
                    ij_encoder_dist = dist_function(lang_i_encoder_emb, lang_j_encoder_emb)
                    ij_decoder_dist = dist_function(lang_i_decoder_emb, lang_j_decoder_emb)
                    encoder_dist += ij_encoder_dist
                    decoder_dist += ij_decoder_dist
                    example_num += 1


    def aggregate_dist_same_lang_different_questions(self, dist_function):
        pass


def main():
    pass