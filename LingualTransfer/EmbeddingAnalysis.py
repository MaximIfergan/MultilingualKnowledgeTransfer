import random

import pandas as pd
import numpy as np
import torch
from Model.MLCBQA_Model import *
import Data.DataPreprocessing as DataPreprocessing

# ===============================      Global Variables:      ===============================

F1_SUCCESS = 0.5

# ===============================      Global functions:      ===============================


def mean_decoder_embedding(embedding):
    result = []
    for layer_index in range(len(embedding[0])):
        layer_mean = torch.zeros_like(embedding[0][layer_index])
        for token_index in range(len(embedding)):
            layer_mean += embedding[token_index][layer_index]
        layer_mean /= len(embedding)
        result.append(layer_mean)
    return result


def mean_encoder_embedding(embedding):
    result = [torch.mean(layer, dim=1) for layer in embedding]
    return result


def cos_similarity(a_embedding, b_embedding):
    result = np.zeros((len(a_embedding)))
    for i in range(len(a_embedding)):
        result[i] = torch.cosine_similarity(torch.flatten(a_embedding[i])[None, :],
                                            torch.flatten(b_embedding[i])[None, :]).item()
    return result


def l2_similarity(a_embedding, b_embedding):
    result = np.zeros((len(a_embedding)))
    for i in range(len(a_embedding)):
        result[i] = torch.cdist(torch.flatten(a_embedding[i])[None, :],
                                torch.flatten(b_embedding[i])[None, :], p=2).item()
    return result


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
        example_num = 0
        df = self.results.loc[self.results['F1'] > F1_SUCCESS]  # only success answers
        df = df.loc[df['Dataset'] != "NQ"]
        ids = list(df["Id"].unique())
        a_lang = list(self.emb_layers[str(ids[0])].keys())[0]
        encoder_dist = np.zeros(len(self.emb_layers[str(ids[0])][a_lang]["encoder_hidden_states"]))
        decoder_dist = np.zeros(len(self.emb_layers[str(ids[0])][a_lang]["decoder_hidden_states"][0]))
        for id in ids:
            langs = list(df.loc[df["Id"] == id]["Language"].unique())
            for i in range(len(langs)):
                lang_i_encoder_emb = mean_encoder_embedding(self.emb_layers[str(id)][langs[i]]["encoder_hidden_states"])
                lang_i_decoder_emb = mean_decoder_embedding(self.emb_layers[str(id)][langs[i]]["decoder_hidden_states"])
                for j in range(i + 1, len(langs)):
                    lang_j_encoder_emb = mean_encoder_embedding(self.emb_layers[str(id)][langs[j]]["encoder_hidden_states"])
                    lang_j_decoder_emb = mean_decoder_embedding(self.emb_layers[str(id)][langs[j]]["decoder_hidden_states"])

                    ij_encoder_dist = dist_function(lang_i_encoder_emb, lang_j_encoder_emb)
                    ij_decoder_dist = dist_function(lang_i_decoder_emb, lang_j_decoder_emb)
                    encoder_dist += ij_encoder_dist
                    decoder_dist += ij_decoder_dist
                    example_num += 1
        encoder_dist = encoder_dist / example_num
        decoder_dist = decoder_dist / example_num
        return encoder_dist, decoder_dist

    def aggregate_dist_same_lang_different_questions(self, dist_function):
        example_num = 0
        df = self.results.loc[self.results['F1'] > F1_SUCCESS]  # only success answers
        df = df.loc[df['Dataset'] != "NQ"]
        ids = list(df["Id"].unique())
        a_lang = list(self.emb_layers[str(ids[0])].keys())[0]
        encoder_dist = np.zeros(len(self.emb_layers[str(ids[0])][a_lang]["encoder_hidden_states"]))
        decoder_dist = np.zeros(len(self.emb_layers[str(ids[0])][a_lang]["decoder_hidden_states"][0]))

        langs = dict()
        for lang in set(DataPreprocessing.MKQA_LANGS + DataPreprocessing.MINTAKA_LANGS):
            langs[lang] = []

        for id in ids:
            for lang in langs:
                if len(langs[lang]) < 200 and lang in self.emb_layers[id]:
                    langs[lang].append(id)

        for lang in langs:
            first_ids = random.choices(langs[lang], k=50)
            second_ids = random.choices(langs[lang], k=50)
            for first_id in first_ids:
                for second_id in second_ids:
                    if first_id == second_id:
                        continue
                    first_encoder_emb = mean_encoder_embedding(
                        self.emb_layers[str(first_id)][lang]["encoder_hidden_states"])
                    first_decoder_emb = mean_decoder_embedding(
                        self.emb_layers[str(first_id)][lang]["decoder_hidden_states"])
                    second_encoder_emb = mean_encoder_embedding(
                        self.emb_layers[str(second_id)][lang]["encoder_hidden_states"])
                    second_decoder_emb = mean_decoder_embedding(
                        self.emb_layers[str(second_id)][lang]["decoder_hidden_states"])

                    ij_encoder_dist = dist_function(first_encoder_emb, second_encoder_emb)
                    ij_decoder_dist = dist_function(first_decoder_emb, second_decoder_emb)
                    encoder_dist += ij_encoder_dist
                    decoder_dist += ij_decoder_dist
                    example_num += 1

        encoder_dist = encoder_dist / example_num
        decoder_dist = decoder_dist / example_num
        return encoder_dist, decoder_dist

    def aggregate_dist_random(self, dist_function):
        example_num = 0
        df = self.results.loc[self.results['F1'] > F1_SUCCESS]  # only success answers
        df = df.loc[df['Dataset'] != "NQ"]
        ids = list(df["Id"].unique())
        a_lang = list(self.emb_layers[str(ids[0])].keys())[0]
        encoder_dist = np.zeros(len(self.emb_layers[str(ids[0])][a_lang]["encoder_hidden_states"]))
        decoder_dist = np.zeros(len(self.emb_layers[str(ids[0])][a_lang]["decoder_hidden_states"][0]))

        first_ids = random.choices(ids, k=80)
        second_ids = random.choices(ids, k=80)
        for first_id in first_ids:
            first_id_lang = random.choice(self.emb_layers[first_id].keys())
            for second_id in second_ids:
                second_id_lang = random.choice(self.emb_layers[second_id].keys())
                first_encoder_emb = mean_encoder_embedding(
                    self.emb_layers[str(first_id)][first_id_lang]["encoder_hidden_states"])
                first_decoder_emb = mean_decoder_embedding(
                    self.emb_layers[str(first_id)][first_id_lang]["decoder_hidden_states"])
                second_encoder_emb = mean_encoder_embedding(
                    self.emb_layers[str(second_id)][second_id_lang]["encoder_hidden_states"])
                second_decoder_emb = mean_decoder_embedding(
                    self.emb_layers[str(second_id)][second_id_lang]["decoder_hidden_states"])

                ij_encoder_dist = dist_function(first_encoder_emb, second_encoder_emb)
                ij_decoder_dist = dist_function(first_decoder_emb, second_decoder_emb)
                encoder_dist += ij_encoder_dist
                decoder_dist += ij_decoder_dist
                example_num += 1

        encoder_dist = encoder_dist / example_num
        decoder_dist = decoder_dist / example_num
        return encoder_dist, decoder_dist


def main():
    print("================ mT5-Base ================")
    pred_dir = "/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-base/predictions.csv"
    with open(
            '/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-base/embedding_layers_mT5-base_new.pkl',
            'rb') as fp:
        embedding_layers = pickle.load(fp)
    ea = EmbeddingAnalysis(embedding_layers, "mT5-base", "Data/Datasets/PreprocessDatasetAllLangs.csv", pred_dir)
    print("aggregate_dist_same_question_different_langs(cos_similarity):")
    print(ea.aggregate_dist_same_question_different_langs(cos_similarity))
    print("aggregate_dist_same_question_different_langs(l2_similarity):")
    print(ea.aggregate_dist_same_question_different_langs(l2_similarity))
    print("aggregate_dist_same_lang_different_questions(cos_similarity):")
    print(ea.aggregate_dist_same_lang_different_questions(cos_similarity))
    print("aggregate_dist_same_lang_different_questions(l2_similarity):")
    print(ea.aggregate_dist_same_lang_different_questions(l2_similarity))
    print("aggregate_dist_random(cos_similarity):")
    print(ea.aggregate_dist_random(cos_similarity))
    print("aggregate_dist_random(l2_similarity)")
    print(ea.aggregate_dist_random(l2_similarity))


    print("================ mT5-Large ================")
    pred_dir = "/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-large/mT5-large-continue/predictions.csv"
    with open('/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-large/mT5-large-continue/embedding_layers_mT5-large.pkl', 'rb') as fp:
        embedding_layers = pickle.load(fp)
    ea = EmbeddingAnalysis(embedding_layers, "mT5-large", "Data/Datasets/PreprocessDatasetAllLangs.csv", pred_dir)
    print("aggregate_dist_same_question_different_langs(cos_similarity):")
    print(ea.aggregate_dist_same_question_different_langs(cos_similarity))
    print("aggregate_dist_same_question_different_langs(l2_similarity):")
    print(ea.aggregate_dist_same_question_different_langs(l2_similarity))
    print("aggregate_dist_same_question_different_langs(cos_similarity):")
    print(ea.aggregate_dist_same_lang_different_questions(cos_similarity))
    print("aggregate_dist_same_question_different_langs(l2_similarity):")
    print(ea.aggregate_dist_same_lang_different_questions(l2_similarity))
    print("aggregate_dist_random(cos_similarity):")
    print(ea.aggregate_dist_random(cos_similarity))
    print("aggregate_dist_random(l2_similarity)")
    print(ea.aggregate_dist_random(l2_similarity))