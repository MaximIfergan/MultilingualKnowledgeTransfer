import json

import jsonlines
import pandas as pd
import spacy
import numpy as np


def print_title(title):
    res = "      " + title + "      "
    while (len(res) < 90):
        res = "=" + res + "="
    print("# " + res)


# print_title("Load models:")

# nlp = spacy.load('en_core_web_sm')
# for name in nlp.pipe_names:
#     print(name)
#     if name == 'lemmatizer':
#         nlp.remove_pipe(name)
#
# # nlp = spacy.blank("en")
# # nlp = spacy.blank("fr")
# # nlp.add_pipe('dbpedia_spotlight')
# # nlp.add_pipe('opentapioca')
# # ner_pipe = nlp.create_pipe("ner")
# # nlp.add_pipe("ner")
# # doc = nlp('Le président des États-Unis appelle Boris Johnson pour décider quoi faire à propos du coronavirus')
# sent = 'The president of USA is calling Boris Johnson to decide what to do about coronavirus'
# # sent = 'who sings love you like theres no tomorrow?'
# # sent = "when did the rams play in the super bowl?"
# doc = nlp(sent)
# # how long did it take the twin towers to be built?
# # doc = nlp("when was the last time the lakers made the playoffs?")
# # doc = nlp("Los Angeles Lakers")
# # doc = nlp('putin was born in russia')
# # doc = nlp('Chris Young is a famous singer')
# # doc = nlp('twin towers')
# # print('Entities', [(ent.text, ent.label_, ent.kb_id_) for ent in doc.ents])
# # for ent in doc.ents:
# #     print(sent[:ent.start_char] + "[START] " + sent[ent.start_char:ent.end_char] + " [END]" + sent[ent.end_char:])
# #
# for token in doc:
#     print(token.text, token.pos_)
#
# for chunk in doc.noun_chunks:
#     print(chunk.text, chunk.start_char, chunk.end_char)
#
# for ent in doc.ents:
#     print(ent)

# for token in doc:
#     print(token.text, token.pos_, token.tag_)
# print(doc.ents[0]._.dbpedia_raw_result)

# import os
# ROOT_PATH = os.path.abspath("")
# FINETUNING_DATA_PATH = ROOT_PATH + "/Data/Datasets/PreprocessDataset.csv"
# OUTPUT_PATH = ROOT_PATH + "/EntityLinking/FinetuningDatasets/Results/finetuning_entities.json"
# CACHE_DIR = ROOT_PATH + "/downloaded_models"
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#
# tokenizer = AutoTokenizer.from_pretrained("facebook/mgenre-wiki", cache_dir=CACHE_DIR)
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mgenre-wiki", cache_dir=CACHE_DIR).eval()
#
# sentences = []
# # sentences.append("[START] Einstein [END] era un fisico tedesco.")
# # sentences.append("[START] The [END] rain is fine")
# # sentences.append("how long did it take the [START] twin towers [END] to be built?")
# # sentences.append("who sings love you like there's no tomorrow? [START] Chris Young [END]")
# sentences.append("how long did it take [START] the twin towers [END] to be built?")
# outputs = model.generate(
#     **tokenizer(sentences, return_tensors="pt"),
#     num_beams=2,
#     num_return_sequences=1,
#     # OPTIONAL: use constrained beam search
#     # prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
# )
#
# res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# print(res)
# import pywikibot
# SITE = pywikibot.Site("en", "wikipedia")
#
# page = pywikibot.Page(SITE, "Israel")
# item = pywikibot.ItemPage.fromPage(page)
# entity_id = item.id
#
# # module load cuda/11.1
# module load torch/1.9-cuda-11.1

# files = ['EntityLinking/FinetuningDatasets/Results/finetuning_entities.json',
#          'EntityLinking/FinetuningDatasets/Results/finetuning_entities2.json',
#          'EntityLinking/FinetuningDatasets/Results/finetuning_entities3.json',
#          "EntityLinking/FinetuningDatasets/Results/finetuning_entities4.json",
#          "EntityLinking/FinetuningDatasets/Results/finetuning_entities5.json"]
#
#
# def merge_JsonFiles(filename):
#     result = list()
#     for f1 in filename:
#         tmp = [json.loads(line) for line in open(f1, 'r', encoding="utf8")]
#         result.extend(tmp)
#
#     with open('EntityLinking/FinetuningDatasets/Results/finetuning_entities_all.json', 'w', encoding="utf8") as output_file:
#         with jsonlines.Writer(output_file) as writer:
#             for i in range(len(result)):
#                 writer.write(result[i])
#
#
# merge_JsonFiles(files)

# string = "hello my name is maxim"
# new_string = "_".join([w.capitalize() for w in string.split()])
# print(new_string)

from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="nkandpa2/pretraining_entities", filename="c4_entity_map.npz", repo_type="dataset", cache_dir="EntityLinking/PretrainingDatasets")
hf_hub_download(repo_id="nkandpa2/pretraining_entities", filename="roots_entity_map.npz", repo_type="dataset", cache_dir="EntityLinking/PretrainingDatasets")