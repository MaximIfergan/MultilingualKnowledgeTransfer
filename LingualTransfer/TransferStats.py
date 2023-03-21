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

# ===============================      Global Variables:      ===============================

MKQA_ENTITIES_TO_PV = "EntityLinking/FinetuningDatasets/Results/MKQA_entities_to_pv.pkl"
MINTAKA_ENTITIES_TO_PV = "EntityLinking/FinetuningDatasets/Results/Mintaka_entities_to_pv.pkl"
CLIENT = Client()


# ===============================      Global Functions:      ===============================


class TransferStats:

    def __init__(self, qa_results_path, id2entities_path, entity2pv_path):
        self.qa_results_path = pd.read_csv(qa_results_path)
        with open(id2entities_path, "rb") as fp:
            self.id2entities = pickle.load(fp)
        with open(entity2pv_path, "rb") as fp:
            self.entity2pv = pickle.load(fp)

    def get_parity_score(self, group_by=None):
        pass

    def get_success_average_pv(self):
        pass


def main():
    count = 0
    df = pd.read_csv("LingualTransfer/Data/query_hebrew_poets_birth_year.csv")
    df["en_views"] = -1
    df["he_views"] = -1
    df["en_q"] = ""
    df["he_q"] = ""
    for index, row in df.iterrows():
        # if count >= 5:
        #     break
        entity_id = row["item"].split("/")[-1]
        df.at[index, "en_views"] = EntityStats.get_daily_average_page_view(entity_id, 'en')
        df.at[index, "he_views"] = EntityStats.get_daily_average_page_view(entity_id, 'he')
        df.at[index, "en_q"] = "What year was " + df.at[index, "itemLabel"] + " born?"
        df.at[index, "he_q"] = "באיזה שנה נולד " + df.at[index, "item_he"] + " ?"
        count += 1
    df.to_csv("LingualTransfer/Data/query_hebrew_poets_birth_year_details.csv")
