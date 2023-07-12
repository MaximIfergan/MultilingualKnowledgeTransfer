import math
import random
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from Model.MLCBQA_Model import *
import Data.DataPreprocessing as DataPreprocessing
from LingualTransfer.CKA import kernel_CKA, linear_CKA

# ===============================      Global Variables:      ===============================

F1_SUCCESS = 0.5

# ===============================      Global functions:      ===============================


def cos_similarity(a_embedding, b_embedding):
    return np.array(torch.cosine_similarity(torch.tensor(a_embedding), torch.tensor(b_embedding), dim=1))
    # result = np.zeros((len(a_embedding)))
    # for i in range(len(a_embedding)):
    #     result[i] = torch.cosine_similarity(torch.flatten(a_embedding[i])[None, :],
    #                                         torch.flatten(b_embedding[i])[None, :]).item()
    # return result


def l2_similarity(a_embedding, b_embedding):
    return np.linalg.norm(a_embedding - b_embedding, axis=1)

    # result = np.zeros((len(a_embedding)))
    # for i in range(len(a_embedding)):
    #     result[i] = torch.cdist(torch.flatten(a_embedding[i])[None, :],
    #                             torch.flatten(b_embedding[i])[None, :], p=2).item()
    # return result


# ====================================      Class:      ====================================


class EmbeddingAnalysis:
    """ this class produce statistics analysis on the success of the model on the different languages """

    def __init__(self, embedding_layers, model_name, data_path, model_predictions):
        self.emb_layers = embedding_layers
        self.model_name = model_name

        predictions = pd.read_csv(model_predictions)
        self.results = pd.read_csv(data_path)
        self.results = self.results.loc[self.results['DataType'] == "dev"]
        self.results["Prediction"] = list(predictions["Generated Text"])
        self.results["F1"] = list(predictions["F1"])
        self.results["EM"] = list(predictions["EM"])
        self.results = self.results.loc[self.results['Dataset'] != "NQ"]

        # self.results = self.results.loc[(self.results['Language'] == "en") | (self.results['Language'] == "ar")]

        self.results["Know"] = 0
        ids = list(self.results.loc[self.results['F1'] > 0.5]["Id"].unique())
        for id in ids:
            self.results.loc[self.results['Id'] == id, 'Know'] = 1

        # self.results = self.results.loc[self.results['Know'] == 1][:1000]  # TODO for debug

        self.results["Id"] = self.results["Id"].apply(lambda x: str(x))

        self.encoder_mean, self.decoder_mean = self.calculate_embedding_mean()
        self.normalize_emb_layers()

    def normalize_emb_layers(self):
        for id in self.emb_layers:
            for lang in self.emb_layers[id]:
                self.emb_layers[id][lang]["encoder_hidden_states"] = \
                    self.normalize_encoder_embedding(self.emb_layers[id][lang]["encoder_hidden_states"])
                self.emb_layers[id][lang]["decoder_hidden_states"] = \
                    self.normalize_decoder_embedding(self.emb_layers[id][lang]["decoder_hidden_states"])

    def calculate_embedding_mean(self):
        first_example = self.results.iloc[0]
        encoder_mean = [torch.zeros_like(layer) for layer
                        in self.emb_layers[first_example["Id"]][first_example["Language"]]["encoder_hidden_states"]]
        decoder_mean = [torch.zeros_like(layer) for layer
                        in self.emb_layers[first_example["Id"]][first_example["Language"]]["decoder_hidden_states"]]
        total_examples = 0
        for id in self.emb_layers:
            for lang in self.emb_layers[id]:
                total_examples += 1
                for i, layer in enumerate(self.emb_layers[id][lang]["encoder_hidden_states"]):
                    encoder_mean[i] += layer
                for i, layer in enumerate(self.emb_layers[id][lang]["decoder_hidden_states"]):
                    decoder_mean[i] += layer
        encoder_mean = [layer / total_examples for layer in encoder_mean]
        decoder_mean = [layer / total_examples for layer in decoder_mean]
        return encoder_mean, decoder_mean

    def normalize_encoder_embedding(self, embedding):
        normalize_embedding = [embedding[i] - self.encoder_mean[i] for i in range(len(embedding))]
        return normalize_embedding

    def normalize_decoder_embedding(self, embedding):
        normalize_embedding = [embedding[i] - self.decoder_mean[i] for i in range(len(embedding))]
        return normalize_embedding

    def same_question_different_langs(self):
        ids = list(self.results["Id"].unique())
        first_group = []
        second_group = []
        for id in ids:
            id_langs = list(self.results.loc[(self.results["Id"] == id) & (self.results['F1'] > 0.5)]["Language"])
            for i in range(len(id_langs)):
                first_group.append((id, id_langs[i]))
                for j in range(i + 1, len(id_langs)):
                    second_group.append((id, id_langs[j]))
        examples_num = min(len(first_group), len(second_group))
        return first_group[:examples_num], second_group[:examples_num]

    def same_lang_different_questions(self):
        first_group = []
        second_group = []
        data_langs = list(self.results["Language"].unique())
        for lang in data_langs:
            lang_ids = list(
                self.results.loc[(self.results["Language"] == lang) & (self.results['F1'] > 0.5)]["Id"].unique())
            lang_ids = [(id, lang) for id in lang_ids]
            first_group += lang_ids[:math.floor(len(lang_ids) / 2)]
            second_group += lang_ids[math.ceil(len(lang_ids) / 2):]
        return first_group, second_group

    def random(self):
        all_q = []
        df = self.results.loc[self.results['F1'] > 0.5]
        for index, row in df.iterrows():
            all_q.append((row["Id"], row["Language"]))
        random.shuffle(all_q)
        first_group = all_q[:math.floor(len(all_q) / 2)]
        second_group = all_q[math.ceil(len(all_q) / 2):]
        return first_group, second_group

    def calculate_distances(self, first_emb_encoder, first_emb_decoder,
                            second_emb_encoder, second_emb_decoder, dist_function):
        # number_of_samples = len(first_group) if number_of_samples == -1 else number_of_samples
        # first_group = first_group[:number_of_samples]
        # second_group = second_group[:number_of_samples]
        # r_id, r_lang = first_group[0]
        #
        # # Init embeddings:
        # first_emb_encoder = [np.zeros((0, self.emb_layers[r_id][r_lang]["encoder_hidden_states"][i].shape[-1]))
        #                      for i in range(len(self.emb_layers[r_id][r_lang]["encoder_hidden_states"]))]
        # second_emb_encoder = [np.zeros((0, self.emb_layers[r_id][r_lang]["encoder_hidden_states"][i].shape[-1]))
        #                       for i in range(len(self.emb_layers[r_id][r_lang]["encoder_hidden_states"]))]
        # first_emb_decoder = [np.zeros((0, self.emb_layers[r_id][r_lang]["decoder_hidden_states"][i].shape[-1]))
        #                      for i in range(len(self.emb_layers[r_id][r_lang]["decoder_hidden_states"]))]
        # second_emb_decoder = [np.zeros((0, self.emb_layers[r_id][r_lang]["decoder_hidden_states"][i].shape[-1]))
        #                       for i in range(len(self.emb_layers[r_id][r_lang]["decoder_hidden_states"]))]
        # for i in range(len(first_group)):
        #     for j in range(len(first_emb_encoder)):
        #         first_emb_encoder[j] = np.concatenate(
        #         (first_emb_encoder[j], self.emb_layers[first_group[i][0]][first_group[i][1]]["encoder_hidden_states"][j]))
        #         second_emb_encoder[j] = np.concatenate(
        #         (second_emb_encoder[j], self.emb_layers[second_group[i][0]][second_group[i][1]]["encoder_hidden_states"][j]))
        #     for j in range(len(first_emb_decoder)):
        #         first_emb_decoder[j] = np.concatenate(
        #         (first_emb_decoder[j], self.emb_layers[first_group[i][0]][first_group[i][1]]["decoder_hidden_states"][j]))
        #         second_emb_decoder[j] = np.concatenate(
        #         (second_emb_decoder[j], self.emb_layers[second_group[i][0]][second_group[i][1]]["decoder_hidden_states"][j]))

        # Calculate distances:
        encoder_distances = [dist_function(first_emb_encoder[i], second_emb_encoder[i])
                             for i in range(len(first_emb_encoder))]
        decoder_distances = [dist_function(first_emb_decoder[i], second_emb_decoder[i])
                             for i in range(len(first_emb_decoder))]
        encoder_mean_dists = [np.mean(layer) for layer in encoder_distances]
        decoder_mean_dists = [np.mean(layer) for layer in decoder_distances]
        encoder_std_dists = [np.std(layer) for layer in encoder_distances]
        decoder_std_dists = [np.std(layer) for layer in decoder_distances]
        return {"encoder": {"mean": encoder_mean_dists, "std": encoder_std_dists},
                "decoder": {"mean": decoder_mean_dists, "std": decoder_std_dists}}

    def get_group_embeddings(self, group, number_of_samples):
        number_of_samples = len(group) if number_of_samples == -1 else number_of_samples
        group = group[:number_of_samples]
        r_id, r_lang = group[0]
        # emb_encoder = [np.zeros((0, self.emb_layers[r_id][r_lang]["encoder_hidden_states"][i].shape[-1]))
        #                      for i in range(len(self.emb_layers[r_id][r_lang]["encoder_hidden_states"]))]
        # emb_decoder = [np.zeros((0, self.emb_layers[r_id][r_lang]["decoder_hidden_states"][i].shape[-1]))
        #                      for i in range(len(self.emb_layers[r_id][r_lang]["decoder_hidden_states"]))]
        emb_encoder = [[] for _ in range(len(self.emb_layers[r_id][r_lang]["encoder_hidden_states"]))]
        emb_decoder = [[] for _ in range(len(self.emb_layers[r_id][r_lang]["decoder_hidden_states"]))]
        # for i in range(len(group)):
        #     for j in range(len(emb_encoder)):
        #         emb_encoder[j] = np.concatenate(
        #         (emb_encoder[j], self.emb_layers[group[i][0]][group[i][1]]["encoder_hidden_states"][j].cpu()))
        #     for j in range(len(emb_decoder)):
        #         emb_decoder[j] = np.concatenate(
        #         (emb_decoder[j], self.emb_layers[group[i][0]][group[i][1]]["decoder_hidden_states"][j].cpu()))
        for i in range(len(group)):
            for j in range(len(emb_encoder)):
                emb_encoder[j].append(self.emb_layers[group[i][0]][group[i][1]]["encoder_hidden_states"][j][0].cpu())
            for j in range(len(emb_decoder)):
                emb_decoder[j].append(self.emb_layers[group[i][0]][group[i][1]]["decoder_hidden_states"][j][0].cpu())
        emb_encoder = [np.stack(layer, axis=0) for layer in emb_encoder]
        emb_decoder = [np.stack(layer, axis=0) for layer in emb_decoder]
        return emb_encoder, emb_decoder

    def plot_layer_dist(self, data, title, dist_function, out=""):
        fig = plt.figure()
        prams = ["#CC4F1B", "#1B2ACC", "#3F7F4C"]
        first_key = list(data.keys())[0]
        x = np.array(list(range(1, len(data[first_key]["mean"]) + 1)))
        for index, type in enumerate(data):
            # plt.errorbar(x, data[type]['mean'], data[type]['std'], label=type)
            plt.plot(x, data[type]['mean'], label=type, color=prams[index])
            plt.plot(x, np.array(data[type]['mean']) + np.array(data[type]['std']), 'v', color=prams[index],
                     markersize=4)
            plt.plot(x, np.array(data[type]['mean']) - np.array(data[type]['std']), '^', color=prams[index],
                     markersize=4)
            # plt.fill_between(x, np.array(data[type]['mean']) + np.array(data[type]['std']),
            #                  np.array(data[type]['mean']), facecolor=prams[index][0], edgecolor=prams[index][0], alpha=0.2)

        plt.legend()
        plt.title(title)
        plt.xlabel("Layers")
        plt.ylabel(dist_function)
        plt.savefig(out + title)

    def plot_all(self, sample_num=5000, out=""):

        print(f"=== Same lang: {self.model_name}")

        first_group, second_group = self.same_lang_different_questions()
        first_emb_encoder, first_emb_decoder = self.get_group_embeddings(first_group, sample_num)
        second_emb_encoder, second_emb_decoder = self.get_group_embeddings(second_group, sample_num)
        same_language_cos = self.calculate_distances(first_emb_encoder, first_emb_decoder,
                                                     second_emb_encoder, second_emb_decoder, cos_similarity)
        same_language_l2 = self.calculate_distances(first_emb_encoder, first_emb_decoder,
                                                    second_emb_encoder, second_emb_decoder, l2_similarity)
        same_language_kernel_CKA = self.calculate_distances(first_emb_encoder, first_emb_decoder,
                                                            second_emb_encoder, second_emb_decoder, kernel_CKA)
        same_language_linear_CKA = self.calculate_distances(first_emb_encoder, first_emb_decoder,
                                                            second_emb_encoder, second_emb_decoder, linear_CKA)

        print(f"=== Same QA {self.model_name}")

        first_group, second_group = self.same_question_different_langs()
        first_emb_encoder, first_emb_decoder = self.get_group_embeddings(first_group, sample_num)
        second_emb_encoder, second_emb_decoder = self.get_group_embeddings(second_group, sample_num)
        same_question_cos = self.calculate_distances(first_emb_encoder, first_emb_decoder,
                                                     second_emb_encoder, second_emb_decoder, cos_similarity)
        same_question_l2 = self.calculate_distances(first_emb_encoder, first_emb_decoder,
                                                    second_emb_encoder, second_emb_decoder, l2_similarity)
        same_question_kernel_CKA = self.calculate_distances(first_emb_encoder, first_emb_decoder,
                                                            second_emb_encoder, second_emb_decoder, kernel_CKA)
        same_question_linear_CKA = self.calculate_distances(first_emb_encoder, first_emb_decoder,
                                                            second_emb_encoder, second_emb_decoder, linear_CKA)

        print(f"=== Random {self.model_name}")

        first_group, second_group = self.random()
        first_emb_encoder, first_emb_decoder = self.get_group_embeddings(first_group, sample_num)
        second_emb_encoder, second_emb_decoder = self.get_group_embeddings(second_group, sample_num)
        random_cos = self.calculate_distances(first_emb_encoder, first_emb_decoder,
                                                     second_emb_encoder, second_emb_decoder, cos_similarity)
        random_l2 = self.calculate_distances(first_emb_encoder, first_emb_decoder,
                                                    second_emb_encoder, second_emb_decoder, l2_similarity)
        random_kernel_CKA = self.calculate_distances(first_emb_encoder, first_emb_decoder,
                                                            second_emb_encoder, second_emb_decoder, kernel_CKA)
        random_linear_CKA = self.calculate_distances(first_emb_encoder, first_emb_decoder,
                                                            second_emb_encoder, second_emb_decoder, linear_CKA)

        # ==== cos similarity: ====

        data = {"Same Language": same_language_cos["encoder"],
                "Same Question": same_question_cos["encoder"],
                "Random": random_cos["encoder"]}
        self.plot_layer_dist(data, f"{self.model_name} encoder layer cos similarity distance", "cos similarity", out)

        data = {"Same Language": same_language_cos["decoder"],
                "Same Question": same_question_cos["decoder"],
                "Random": random_cos["decoder"]}
        self.plot_layer_dist(data, f"{self.model_name} decoder layer cos similarity distance", "cos similarity", out)

        # ==== L2: ====

        data = {"Same Language": same_language_l2["encoder"],
                "Same Question": same_question_l2["encoder"],
                "Random": random_l2["encoder"]}
        self.plot_layer_dist(data, f"{self.model_name} encoder layer L2 distance", "L2", out)

        data = {"Same Language": same_language_l2["decoder"],
                "Same Question": same_question_l2["decoder"],
                "Random": random_l2["decoder"]}
        self.plot_layer_dist(data, f"{self.model_name} decoder layer L2 distance", "L2", out)

        # ==== kernel_CKA: ====

        data = {"Same Language": same_language_kernel_CKA["encoder"],
                "Same Question": same_question_kernel_CKA["encoder"],
                "Random": random_kernel_CKA["encoder"]}
        self.plot_layer_dist(data, f"{self.model_name} encoder layer kernel_CKA distance", "kernel CKA", out)

        data = {"Same Language": same_language_kernel_CKA["decoder"],
                "Same Question": same_question_kernel_CKA["decoder"],
                "Random": random_kernel_CKA["decoder"]}
        self.plot_layer_dist(data, f"{self.model_name} decoder layer kernel CKA distance", "kernel CKA", out)

        # ==== linear_CKA: ====

        data = {"Same Language": same_language_linear_CKA["encoder"],
                "Same Question": same_question_linear_CKA["encoder"],
                "Random": random_linear_CKA["encoder"]}
        self.plot_layer_dist(data, f"{self.model_name} encoder layer linear CKA distance", "linear_CKA", out)

        data = {"Same Language": same_language_linear_CKA["decoder"],
                "Same Question": same_question_linear_CKA["decoder"],
                "Random": random_linear_CKA["decoder"]}
        self.plot_layer_dist(data, f"{self.model_name} decoder layer linear CKA distance", "linear_CKA", out)

    # ========================== Old Methods: ==========================

    def aggregate_dist_same_question_different_langs(self, dist_function):
        ids = list(self.results["Id"].unique())
        a_lang = list(self.emb_layers[ids[0]].keys())[0]
        encoder_dists = [[] for _ in range(len(self.emb_layers[ids[0]][a_lang]["encoder_hidden_states"]))]
        decoder_dists = [[] for _ in range(len(self.emb_layers[ids[0]][a_lang]["decoder_hidden_states"]))]
        for id in ids:
            langs = list(self.results.loc[self.results["Id"] == id]["Language"].unique())
            for i in range(len(langs)):
                lang_i_encoder_emb = self.emb_layers[id][langs[i]]["encoder_hidden_states"]
                lang_i_decoder_emb = self.emb_layers[id][langs[i]]["decoder_hidden_states"]
                for j in range(i + 1, len(langs)):
                    lang_j_encoder_emb = self.emb_layers[id][langs[j]]["encoder_hidden_states"]
                    lang_j_decoder_emb = self.emb_layers[id][langs[j]]["decoder_hidden_states"]
                    ij_encoder_dist = dist_function(lang_i_encoder_emb, lang_j_encoder_emb)
                    ij_decoder_dist = dist_function(lang_i_decoder_emb, lang_j_decoder_emb)
                    for i_lyr, layer_dist in enumerate(ij_encoder_dist):
                        encoder_dists[i_lyr].append(layer_dist)
                    for i_lyr, layer_dist in enumerate(ij_decoder_dist):
                        decoder_dists[i_lyr].append(layer_dist)
        encoder_mean_dists = [np.mean(layer).item() for layer in encoder_dists]
        encoder_std_dists = [np.std(layer).item() for layer in encoder_dists]
        decoder_mean_dists = [np.mean(layer).item() for layer in decoder_dists]
        decoder_std_dists = [np.std(layer).item() for layer in decoder_dists]
        return {"encoder": {"mean": encoder_mean_dists, "std": encoder_std_dists},
                "decoder": {"mean": decoder_mean_dists, "std": decoder_std_dists}}

    def aggregate_dist_same_lang_different_questions(self, dist_function):
        ids = list(self.results["Id"].unique())
        a_lang = list(self.emb_layers[ids[0]].keys())[0]
        encoder_dists = [[] for _ in range(len(self.emb_layers[ids[0]][a_lang]["encoder_hidden_states"]))]
        decoder_dists = [[] for _ in range(len(self.emb_layers[ids[0]][a_lang]["decoder_hidden_states"]))]

        langs = dict()
        for lang in list(self.results["Language"].unique()):
            langs[lang] = []

        for id in ids:
            for lang in langs:
                if len(langs[lang]) < 200 and lang in self.emb_layers[id]:
                    langs[lang].append(id)

        for lang in langs:
            if not langs[lang]:
                continue
            first_ids = random.choices(langs[lang], k=20)  # k = 20
            second_ids = random.choices(langs[lang], k=20)  # k = 20
            for first_id in first_ids:
                for second_id in second_ids:
                    if first_id == second_id:
                        continue
                    first_encoder_emb = self.emb_layers[first_id][lang]["encoder_hidden_states"]
                    first_decoder_emb = self.emb_layers[first_id][lang]["decoder_hidden_states"]
                    second_encoder_emb = self.emb_layers[second_id][lang]["encoder_hidden_states"]
                    second_decoder_emb = self.emb_layers[second_id][lang]["decoder_hidden_states"]
                    ij_encoder_dist = dist_function(first_encoder_emb, second_encoder_emb)
                    ij_decoder_dist = dist_function(first_decoder_emb, second_decoder_emb)
                    for i_lyr, layer_dist in enumerate(ij_encoder_dist):
                        encoder_dists[i_lyr].append(layer_dist)
                    for i_lyr, layer_dist in enumerate(ij_decoder_dist):
                        decoder_dists[i_lyr].append(layer_dist)
        encoder_mean_dists = [np.mean(layer).item() for layer in encoder_dists]
        encoder_std_dists = [np.std(layer).item() for layer in encoder_dists]
        decoder_mean_dists = [np.mean(layer).item() for layer in decoder_dists]
        decoder_std_dists = [np.std(layer).item() for layer in decoder_dists]
        return {"encoder": {"mean": encoder_mean_dists, "std": encoder_std_dists},
                "decoder": {"mean": decoder_mean_dists, "std": decoder_std_dists}}

    def aggregate_dist_random(self, dist_function):
        ids = list(self.results["Id"].unique())
        a_lang = list(self.emb_layers[ids[0]].keys())[0]
        encoder_dists = [[] for _ in range(len(self.emb_layers[ids[0]][a_lang]["encoder_hidden_states"]))]
        decoder_dists = [[] for _ in range(len(self.emb_layers[ids[0]][a_lang]["decoder_hidden_states"]))]

        first_ids = random.choices(ids, k=50)  # k = 50
        second_ids = random.choices(ids, k=50)  # k = 50
        for first_id in first_ids:
            first_id_lang = random.choice(list(self.emb_layers[first_id].keys()))
            for second_id in second_ids:
                second_id_lang = random.choice(list(self.emb_layers[second_id].keys()))
                if second_id == first_id and first_id_lang == second_id_lang:
                    continue

                first_encoder_emb = self.emb_layers[first_id][first_id_lang]["encoder_hidden_states"]
                first_decoder_emb = self.emb_layers[first_id][first_id_lang]["decoder_hidden_states"]
                second_encoder_emb = self.emb_layers[second_id][second_id_lang]["encoder_hidden_states"]
                second_decoder_emb = self.emb_layers[second_id][second_id_lang]["decoder_hidden_states"]
                ij_encoder_dist = dist_function(first_encoder_emb, second_encoder_emb)
                ij_decoder_dist = dist_function(first_decoder_emb, second_decoder_emb)
                for i_lyr, layer_dist in enumerate(ij_encoder_dist):
                    encoder_dists[i_lyr].append(layer_dist)
                for i_lyr, layer_dist in enumerate(ij_decoder_dist):
                    decoder_dists[i_lyr].append(layer_dist)

        encoder_mean_dists = [np.mean(layer).item() for layer in encoder_dists]
        encoder_std_dists = [np.std(layer).item() for layer in encoder_dists]
        decoder_mean_dists = [np.mean(layer).item() for layer in decoder_dists]
        decoder_std_dists = [np.std(layer).item() for layer in decoder_dists]
        return {"encoder": {"mean": encoder_mean_dists, "std": encoder_std_dists},
                "decoder": {"mean": decoder_mean_dists, "std": decoder_std_dists}}


def main():
    # # Flatten base decoder embeddings:
    # with open("/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-base/embedding_layers_all.pkl", 'rb') as fp:
    #     embedding_layers = pickle.load(fp)
    # for id in embedding_layers:
    #     for lang in embedding_layers[id]:
    #         for i in range(len(embedding_layers[id][lang]["decoder_hidden_states"])):
    #             assert embedding_layers[id][lang]["decoder_hidden_states"][i].shape[0] == 2
    #             embedding_layers[id][lang]["decoder_hidden_states"][i] = torch.flatten(embedding_layers[id][lang]["decoder_hidden_states"][i])[None, :]
    # with open('/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-base/embedding_layers_all_flatten.pkl', 'wb') as fp:
    #     pickle.dump(embedding_layers, fp)
    #
    # # Flatten large decoder embeddings:
    # with open("/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-large/mT5-large-continue/embedding_layers_all.pkl", 'rb') as fp:
    #     embedding_layers = pickle.load(fp)
    # for id in embedding_layers:
    #     for lang in embedding_layers[id]:
    #         for i in range(len(embedding_layers[id][lang]["decoder_hidden_states"])):
    #             assert embedding_layers[id][lang]["decoder_hidden_states"][i].shape[0] == 2
    #             embedding_layers[id][lang]["decoder_hidden_states"][i] = torch.flatten(embedding_layers[id][lang]["decoder_hidden_states"][i])[None, :]
    # with open('/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-large/mT5-large-continue/embedding_layers_all_flatten.pkl', 'wb') as fp:
    #     pickle.dump(embedding_layers, fp)

    # ============================= Analysis Result Last meeting ===============================

    # print("Start base:")
    # pred_dir = "/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-base/predictions.csv"
    # with open("/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-base/embedding_layers_all_flatten.pkl", 'rb') as fp:
    #     embedding_layers = pickle.load(fp)
    # ea = EmbeddingAnalysis(embedding_layers, "mT5-base", "Data/Datasets/PreprocessDatasetAllLangs.csv", pred_dir)
    # ea.plot_all(out="plots/mT5-base/")
    # print("End base:")

    print("Start large:")
    pred_dir = "/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-large/mT5-large-continue/predictions.csv"
    with open("/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-large/mT5-large-continue/embedding_layers_all_flatten.pkl", 'rb') as fp:
        embedding_layers = pickle.load(fp)
    ea = EmbeddingAnalysis(embedding_layers, "mT5-large", "Data/Datasets/PreprocessDatasetAllLangs.csv", pred_dir)
    ea.plot_all(out="plots/mT5-large/")
    print("End large:")

    # ============================= flatten decoder embeddings  ===============================

    # with open("embedding_layers_test.pkl", 'rb') as fp:
    #     embedding_layers = pickle.load(fp)
    # for id in embedding_layers:
    #     for lang in embedding_layers[id]:
    #         for i in range(len(embedding_layers[id][lang]["decoder_hidden_states"])):
    #             assert embedding_layers[id][lang]["decoder_hidden_states"][i].shape[0] == 2
    #             embedding_layers[id][lang]["decoder_hidden_states"][i] = torch.flatten(embedding_layers[id][lang]["decoder_hidden_states"][i])[None, :]
    # with open('embedding_layers_test_fix.pkl', 'wb') as fp:
    #     pickle.dump(embedding_layers, fp)
    # exit(0)

    # # ============================= test result on local computer  ===============================
    # pred_dir = "Model/SavedModels/mT5-base-4-ep/predictions.csv"
    # with open("embedding_layers_test_fix.pkl", 'rb') as fp:
    #     embedding_layers = pickle.load(fp)
    # ea = EmbeddingAnalysis(embedding_layers, "mT5-large", "Data/Datasets/PreprocessDatasetAllLangs.csv", pred_dir)
    # ea.plot_all()
