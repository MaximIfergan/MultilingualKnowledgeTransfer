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

WIKI_PATH = "/cs/labs/oabend/maximifergan/MultilingualKnowledgeTransfer/EntityLinking/PretrainingDatasets/datasets--nkandpa2--pretraining_entities/snapshots/550b4b11a5ac147bf261ff150a65b98b01469b3f/wikipedia_entity_map.npz"
training_entities = np.load(WIKI_PATH)
