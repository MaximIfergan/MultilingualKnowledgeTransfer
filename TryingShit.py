# import json
# import re
import os
import torch
# import Data.DataPreprocessing as DataPreprocessing
# import jsonlines
# import pandas as pd
# import spacy
import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import tensorflow as tf
from transformers import MT5ForConditionalGeneration, MT5Tokenizer


def print_title(title):
    res = "      " + title + "      "
    while (len(res) < 90):
        res = "=" + res + "="
    print("# " + res)


# print_title("save embeddings:")
# output_dir = "Model/SavedModels"
# model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
# path = os.path.join(output_dir, "model_files")
# model.save_pretrained(path)

lst = ["What is George Rankin's occupation?",
       "What is Henry Feilden's occupation?",
       "What is Eleanor Davis's occupation?",
       "What is Alexander Rinnooy Kan's occupation?",
       "Who was the composer of Full House?",
       "Who was the composer of Champion?",
       "What is the capital of Netherlands?",
       "What is the capital of Merthyr Tydfil County Borough?",
       "איפה נולד בנימין נתניהו?",
       "מה העיסוק של אביגדור ליברמן",
       ]

for snt in lst:
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    t = tokenizer.batch_encode_plus([snt], return_tensors="pt")
    print(t["input_ids"].squeeze().shape[0])

# # ==============================      Fix entity2pv dict:      ==============================
#
# with open("EntityLinking/FinetuningDatasets/Results/entity2pv.pkl", "rb") as fp:
#     entity2pv = pickle.load(fp)
#
# for lang in DataPreprocessing.FINETUNING_LANGS:
#     for q in entity2pv[lang]:
#         if entity2pv[lang][q][0] == (-1, -1):
#             entity2pv[lang][q] = (-1, entity2pv[lang][q][1])
#         if entity2pv[lang][q][1] == (-1, -1):
#             entity2pv[lang][q] = (entity2pv[lang][q][0], -1)
#         if isinstance(entity2pv[lang][q][1], str):
#             entity2pv[lang][q] = (entity2pv[lang][q][1], entity2pv[lang][q][0])
#
# with open("EntityLinking/FinetuningDatasets/Results/entity2pv1.pkl", "wb") as fp:
#     pickle.dump(entity2pv, fp)


# # ===================      Merge all the mapping to pv dictionaries:      ===================
#
# entity2pv = dict()
#
# # === MKQA ===
# with open("EntityLinking/FinetuningDatasets/Results/MKQA_entities_to_pv.pkl", "rb") as fp:
#     mkqa_dict = pickle.load(fp)
#
# # === Mintaka ===
# with open("EntityLinking/FinetuningDatasets/Results/Mintaka_entities_to_pv.pkl", "rb") as fp:
#     mintaka_dict = pickle.load(fp)
#
# # === PopQA ===
# with open("EntityLinking/FinetuningDatasets/Results/PopQA_entities_to_pv.pkl", "rb") as fp:
#     popqa_dict = pickle.load(fp)
#
# for lang in DataPreprocessing.FINETUNING_LANGS:
#     entity2pv[lang] = dict()
#     entity2pv[lang].update(mkqa_dict[lang])
#     entity2pv[lang].update(mintaka_dict[lang])
#     entity2pv[lang].update(popqa_dict[lang])
#
# with open("EntityLinking/FinetuningDatasets/Results/entity2pv.pkl", "wb") as fp:
#     pickle.dump(entity2pv, fp)


# =============================      Turn PopQA to map pv:      =============================

# data = pd.read_csv("EntityLinking/FinetuningDatasets/Results/PopQA_pv_stats.csv", delimiter=",")
#
# entity2pv = dict()
# for lang in DataPreprocessing.FINETUNING_LANGS:
#     entity2pv[lang] = dict()
#
# for index, row in data.iterrows():
#     for lang in DataPreprocessing.FINETUNING_LANGS:
#         s_id = row["s_uri"].split("/")[-1]  # subject id
#         o_id = row["o_uri"].split("/")[-1]  # object id
#         if str(row[lang + "_s_pv"]) != "-1":
#             entity2pv[lang][s_id] = (row[lang + "_s_pv"], row[lang + "_s_label"])
#         if str(row[lang + "_o_pv"]) != "-1":
#             entity2pv[lang][o_id] = (row[lang + "_o_pv"], row[lang + "_o_label"])
#
# with open("EntityLinking/FinetuningDatasets/Results/PopQA_entities_to_pv.pkl", "wb") as fp:
#     pickle.dump(entity2pv, fp)

# ============================      Build id2entities dict:      ============================

# id2entities = dict()
# # ==== MKQA ===
#
# for line in open("Data/Datasets/MKQA/MKQA_Linked_Entities.json", 'r', encoding='utf8'):
#     qa_entities = json.loads(line)
#     for entity in qa_entities["q_entities"] + qa_entities["a_entities"]:
#         if not qa_entities["q_entities"] and not qa_entities["a_entities"]:
#             id2entities[str(qa_entities["Id"])] = None
#         q_ids = [ent[1] for ent in qa_entities["q_entities"]]
#         a_ids = [ent[1] for ent in qa_entities["a_entities"]]
#         id2entities[str(qa_entities["Id"])] = {"q_entities": q_ids, "a_entities": a_ids}
#
# # === NQ ===
#
# for line in open("Data/Datasets/NQ/NQ_Linked_Entities.json", 'r', encoding='utf8'):
#     qa_entities = json.loads(line)
#     for entity in qa_entities["q_entities"] + qa_entities["a_entities"]:
#         if not qa_entities["q_entities"] and not qa_entities["a_entities"]:
#             id2entities[str(qa_entities["Id"])] = None
#         q_ids = [ent[1] for ent in qa_entities["q_entities"]]
#         a_ids = [ent[1] for ent in qa_entities["a_entities"]]
#         id2entities[str(qa_entities["Id"])] = {"q_entities": q_ids, "a_entities": a_ids}
#
# # === Mintaka ===
#
# for dtype in ["dev", "test", "train"]:
#     with open(f"Data/Datasets/Mintaka/mintaka_{dtype}.json", 'r', encoding='utf-8') as fp:
#         data = json.load(fp)
#         for qa in data:
#             if qa['answer']['answer'] is not None and len(qa['answer']['answer']) > 1:
#                 id2entities[str(qa["id"])] = None
#                 continue
#             id2entities[str(qa["id"])] = dict()
#             id2entities[str(qa["id"])]["q_entities"] = []
#             id2entities[str(qa["id"])]["a_entities"] = []
#             for entity in qa["questionEntity"]:
#                 if entity["entityType"] != "entity" or entity["name"] is None or \
#                         not re.match("^Q[1-9]+", entity["name"]):
#                     continue
#                 id2entities[str(qa["id"])]["q_entities"].append(entity["name"])
#
#             if qa['answer']['answerType'] == "entity" and qa['answer']['answer'] is not None:
#                 answer_entity = qa['answer']['answer'][0]
#                 if re.match("^Q[1-9]+", answer_entity["name"]):
#                     id2entities[str(qa["id"])]["q_entities"].append(answer_entity["name"])
#
#             if not id2entities[str(qa["id"])]["q_entities"] and not id2entities[str(qa["id"])]["a_entities"]:
#                 id2entities[str(qa["id"])] = None
#
# # === PopQA ===
#
# data = pd.read_csv("Data/Datasets/POPQA/popQA.tsv", sep='\t')
# for index, row in data.iterrows():
#     s_id = row["s_uri"].split("/")[-1]  # subject id
#     o_id = row["o_uri"].split("/")[-1]  # object id
#     id2entities[row["id"]] = dict()
#     id2entities[row["id"]]["q_entities"] = [s_id]
#     id2entities[row["id"]]["a_entities"] = [o_id]
#
# with open("id2entities.pkl", "wb") as fp:
#     pickle.dump(id2entities, fp)

# ===========     Load the Pretraining datasets maps to number of appearance:     ===========

# WIKI_PATH = "EntityLinking/PretrainingDatasets/wikipedia_entity_map.npz"
# training_entities = np.load(WIKI_PATH)

# ===============      Evaluate running time of different pv extraction:      ===============

# import time
# import pywikibot.data.api as api
# import pywikibot
#
# num = 1
# start_time = time.perf_counter()
# for i in range(num):
#     subject = 'Python (programming language)'
#     url = 'https://en.wikipedia.org/w/api.php'
#     params = {
#         'action': 'query',
#         'format': 'json',
#         'titles': subject,
#         'prop': 'pageviews',
#     }
#     response = requests.get(url, params=params)
#     data = response.json()
#     a = data["query"]["pages"]["23862"]["pageviews"]
#
# end_time = time.perf_counter()
#
# # Calculate the running time and print it to the console
# running_time = end_time - start_time
# print(f"Running time First: {running_time:.4f} seconds")
#
# start_time = time.perf_counter()
# for i in range(num):
#     site = pywikibot.Site("en", "wikipedia")
#     page = pywikibot.Page(site, "Python (programming language)")
#     subject = 'Python (programming language)'
#     url = 'https://en.wikipedia.org/w/api.php'
#
#     req = api.Request(site=url, parameters={'action': 'query',
#                                              'titles': subject,
#                                              'prop': 'pageviews'})
#     page_view_stats = req.submit()['query']['pages'][str(page.pageid)]['pageviews']
#
# end_time = time.perf_counter()
#
# # Calculate the running time and print it to the console
# running_time = end_time - start_time
# print(f"Running time secound: {running_time:.4f} seconds")
#
# # Running time First: 383.3474 seconds
# # Running time secound: 481.7285 seconds

# ============================      Load pickle dictionary:      ============================


# with open("EntityLinking/FinetuningDatasets/Results/PopQA_entities_to_pv.pkl", "rb") as fp:
#     b = pickle.load(fp)
# print()
