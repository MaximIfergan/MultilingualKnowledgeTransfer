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
import EntityLinking.FinetuningDatasets.EntityStats as EntityStats
import matplotlib as mpl
mpl.use('TkAgg')  # !IMPORTANT

# ===============================      Global Variables:      ===============================

MKQA_ENTITIES_TO_PV = "EntityLinking/FinetuningDatasets/Results/MKQA_entities_to_pv.pkl"
MINTAKA_ENTITIES_TO_PV = "EntityLinking/FinetuningDatasets/Results/Mintaka_entities_to_pv.pkl"
CLIENT = Client()


# ===============================      Global Functions:      ===============================


class TransferStats:

    def __init__(self, qa_results_path, id2entities_path, entity2pv_path):
        df = pd.read_csv(qa_results_path)
        self.qa_results_path = df.loc[df['Dataset'] != "NQ"]  # Only parallel datasets
        with open(id2entities_path, "rb") as fp:
            self.id2entities = pickle.load(fp)
        with open(entity2pv_path, "rb") as fp:
            self.entity2pv = pickle.load(fp)

    def get_parity_score(self, filter_dataset=None):
        df = self.qa_results_path
        if filter_dataset:
            df = self.qa_results_path[self.qa_results_path[filter_dataset["col"]] == filter_dataset["value"]]
        only_correct_answers = df[df['F1'] > 0.5]
        correct_ids = set(only_correct_answers["Id"])
        number_of_missed_answers = 0
        total = 0
        for index, row in df.iterrows():
            if row['F1'] > 0.5:  # Correct Answer
                continue
            total += 1
            if row['Id'] in correct_ids:
                number_of_missed_answers += 1
        return round(number_of_missed_answers / total, 4)

    def get_success_average_pv(self):
        df = self.qa_results_path[self.qa_results_path['F1'] > 0.5]
        average_dict = {}
        count_id = 0
        count_pv = 0
        for lang in DataPreprocessing.FINETUNING_LANGS:
            average_dict[lang] = []
        for index, row in df.iterrows():
            if row["Id"] in self.id2entities and self.id2entities[str(row["Id"])]:
                entities = self.id2entities[str(row["Id"])]
            else:
                count_id += 1
                continue
            average_pv = []
            for entity in entities["q_entities"] + entities["a_entities"]:
                if entity in self.entity2pv[row["Language"]] and self.entity2pv[row["Language"]][entity][1] != -1:
                    average_pv.append(self.entity2pv[row["Language"]][entity][1])
                else:
                    count_pv += 1
            if not average_pv:
                continue
            average_dict[row["Language"]].append(sum(average_pv) / len(average_pv))
        for lang in DataPreprocessing.FINETUNING_LANGS:
            # average_dict[lang] = sum(average_dict[lang]) / len(average_dict[lang])
            # average_dict[lang] = min(average_dict[lang])
            average_dict[lang].sort()
            average_dict[lang] = average_dict[lang][:500]
            plt.hist(average_dict[lang], 15)
            plt.title(lang)
            plt.show()
        print(count_pv)
        print(count_id)
        return average_dict


def main():
    # count = 0
    # df = pd.read_csv("LingualTransfer/Data/query_hebrew_poets_birth_year.csv")
    # df["en_views"] = -1
    # df["he_views"] = -1
    # df["en_q"] = ""
    # df["he_q"] = ""
    # for index, row in df.iterrows():
    #     # if count >= 5:
    #     #     break
    #     entity_id = row["item"].split("/")[-1]
    #     df.at[index, "en_views"] = EntityStats.get_daily_average_page_view(entity_id, 'en')
    #     df.at[index, "he_views"] = EntityStats.get_daily_average_page_view(entity_id, 'he')
    #     df.at[index, "en_q"] = "What year was " + df.at[index, "itemLabel"] + " born?"
    #     df.at[index, "he_q"] = "באיזה שנה נולד " + df.at[index, "item_he"] + " ?"
    #     count += 1
    # df.to_csv("LingualTransfer/Data/query_hebrew_poets_birth_year_details.csv")
    qa_results_path = "Model/SavedModels/mT5-base-6-epochs/validation_set_with_results.csv"
    id2entities_path = "EntityLinking/FinetuningDatasets/Results/id2entities.pkl"
    entity2pv_path = "EntityLinking/FinetuningDatasets/Results/entity2pv1.pkl"
    ts = TransferStats(qa_results_path, id2entities_path, entity2pv_path)
    # print(ts.get_parity_score({"col": "Dataset", "value": "Mintaka"}))
    # print(ts.get_parity_score({"col": "Dataset", "value": "MKQA"}))
    # print(ts.get_parity_score())
    ts.get_success_average_pv()
