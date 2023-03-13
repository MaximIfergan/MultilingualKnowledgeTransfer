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

# Try

data = pd.read_csv("backup.csv", delimiter=",")
print("123")