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

WIKI_PATH = "EntityLinking/PretrainingDatasets/wikipedia_entity_map.npz"
training_entities = np.load(WIKI_PATH)

