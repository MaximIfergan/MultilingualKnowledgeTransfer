import json
import jsonlines
import pandas as pd
import spacy
import numpy as np
import matplotlib.pyplot as plt

def print_title(title):
    res = "      " + title + "      "
    while (len(res) < 90):
        res = "=" + res + "="
    print("# " + res)

# print_title("Load models:")

# WIKI_PATH = "EntityLinking/PretrainingDatasets/wikipedia_entity_map.npz"
# training_entities = np.load(WIKI_PATH)


# import pywikibot
#
# site = pywikibot.Site("en", "wikipedia")
# page = pywikibot.Page(site, "Israel")
# item = pywikibot.ItemPage.fromPage(page)
# item_dict = item.get()
# print(item_dict["labels"])  # Prints the entities name in different languages
# clm_dict = item_dict["claims"]   # Statements:
# clm_list = clm_dict["P361"]   # Part-of:
# for clm in clm_list:  # Print entities that are part of
#     print("Q" + str(clm.toJSON()["mainsnak"]["datavalue"]["value"]["numeric-id"]))

import requests
#
# # resp = requests.get('https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/Albert_Einstein/daily/2015100100/2015103100')
# # data = resp.json()
# # print(data['items'][0]['views'])


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
#
# import sys
#
# a = dict()
# try:
#     b = a["5"]
# except Exception as e:
#     sys.stderr.write(str(e))

import pickle

with open("EntityLinking/FinetuningDatasets/Results/entities_to_pv (copy).pkl", "rb") as fp:
    b = pickle.load(fp)
print()

#  ===============================  another script for multi process   =======================
#
# #!/bin/bash
# #SBATCH -N 1      # nodes requested
# #SBATCH -n 10      # tasks requested
# #SBATCH --cpus-per-task=2  # cpus-per-task
# #SBATCH --mem=100000  # memory in Mb
# #SBATCH -o JobsOutput/Job_STDOUT_%j  # send stdout to outfile
# #SBATCH -e JobsOutput/Job_STDERR_%j  # send stderr to errfile
# #SBATCH -t 0-15:00:00  # time requested in hour:minute:second
# #SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
# #SBATCH --mail-user=maxim.ifergan@mail.huji.ac.il
# #SBATCH --gres=gpu:1,vmem:6g
#
# srun -n 10 bash run_foo.sh
#
# source /cs/labs/oabend/maximifergan/MultilingualKnowledgeTransfer/venv/bin/activate.csh
# /cs/labs/oabend/maximifergan/MultilingualKnowledgeTransfer/venv/bin/python /cs/labs/oabend/maximifergan/MultilingualKnowledgeTransfer/main.py