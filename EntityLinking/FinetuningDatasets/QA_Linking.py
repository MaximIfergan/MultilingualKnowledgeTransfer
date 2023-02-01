import json
import jsonlines
import pandas as pd
import spacy
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pywikibot
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# ===============================      Global Variables:      ===============================

# ROOT_PATH = os.path.abspath("")
FINETUNING_DATA_PATH = "/cs/labs/oabend/maximifergan/MultilingualKnowledgeTransfer/Data/Datasets/PreprocessDataset.csv"
OUTPUT_PATH = "/cs/labs/oabend/maximifergan/MultilingualKnowledgeTransfer/EntityLinking/FinetuningDatasets/Results/finetuning_entities5.json"
CACHE_DIR = "/cs/labs/oabend/maximifergan/MultilingualKnowledgeTransfer/downloaded_models"
START_TOKEN = "[START]"
END_TOKEN = "[END]"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NLP = spacy.load('en_core_web_sm')
NLP.remove_pipe("lemmatizer")
NLP.remove_pipe("ner")
# NLP.add_pipe('dbpedia_spotlight')  # pip install spacy-dbpedia-spotlight
SITE = pywikibot.Site("en", "wikipedia")
STOP_WORDS = set(stopwords.words('english'))

# ===============================      Global Functions:      ===============================


def link_qa_pair(question, answer, model, tokenizer):

    q_doc = NLP(question)
    q_entities, a_entities = [], []

    # # Using 'dbpedia_spotlight':
    # q_entities = [ent._.dbpedia_raw_result for ent in q_doc.ents]
    # a_entities = [ent._.dbpedia_raw_result for ent in a_doc.ents]

    # Tag question entities:
    for chunk in q_doc.noun_chunks:
        if chunk.text in STOP_WORDS or chunk.text.startswith("how") or chunk.text.startswith("what"):
            continue
        inf_sent = str(question[:chunk.start_char]) + START_TOKEN + " " + \
                   str(question[chunk.start_char:chunk.end_char]) + " " + END_TOKEN \
                   + str(question[chunk.end_char:])

        outputs = model.generate(**tokenizer(inf_sent, return_tensors="pt").to(DEVICE), num_beams=2, num_return_sequences=1)
        entity = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if not entity:
            continue
        tmp = entity[0].split(" >> ")
        if len(tmp) == 2 and tmp[0] != "":
            entity_name, entity_lang = tmp
            try:
                page = pywikibot.Page(SITE, entity_name)
                item = pywikibot.ItemPage.fromPage(page)
                entity_id = item.id
                q_entities.append((entity_name, entity_id))
                # print(f"\n ========== text: {chunk.text} , entity: {entity_name}\n")
            except pywikibot.exceptions.NoPageError:
                # print(f"\nPage {entity_name} not found ")
                pass
            except pywikibot.exceptions.InvalidTitleError:
                pass

    # Tag answer entities:
    inf_sent = START_TOKEN + " " + str(answer) + " " + END_TOKEN
    outputs = model.generate(**tokenizer(inf_sent, return_tensors="pt").to(DEVICE), num_beams=2, num_return_sequences=1)
    entity = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if entity:
        tmp = entity[0].split(" >> ")
        if len(tmp) == 2 and tmp[0] != "":
            entity_name, entity_lang = entity[0].split(" >> ")
            try:
                page = pywikibot.Page(SITE, entity_name)
                item = pywikibot.ItemPage.fromPage(page)
                entity_id = item.id
                a_entities.append((entity_name, entity_id))
                # print(f"\n ========== text: {answer} , entity: {item.aliases._data['en'][0]}")
                # print(f"\n ========== text: {answer} , entity: {entity_name}")
            except pywikibot.exceptions.NoPageError:
                # print(f"\nPage {entity_name} not found ")
                pass
            except pywikibot.exceptions.InvalidTitleError:
                pass

    return {'q_entities': q_entities,
            'a_entities': a_entities}


def link_finetuning_dataset(model, tokenizer, input_path=FINETUNING_DATA_PATH, output_path=OUTPUT_PATH):
    df = pd.read_csv(input_path)
    data = df[["Question", "Answer", "Id", "Language", "Dataset"]].to_numpy()
    flag = True  # Start from the QA when the bug stops
    with open(output_path, 'w') as outfile:
        with jsonlines.Writer(outfile) as writer:
            for i in tqdm(range(data.shape[0])):
                if str(data[i][2]) == "-963872017918043146":
                    flag = False
                if flag or data[i][3] != "en" or data[i][4] == "Mintaka":
                    continue
                qa_entities = link_qa_pair(data[i][0], data[i][1], model, tokenizer)
                qa_entities["Id"] = data[i][2]
                writer.write(qa_entities)


def main():
    tokenizer = AutoTokenizer.from_pretrained("facebook/mgenre-wiki", cache_dir=CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mgenre-wiki", cache_dir=CACHE_DIR).eval().to(DEVICE)
    link_finetuning_dataset(model, tokenizer)
    # tokenizer = AutoTokenizer.from_pretrained("facebook/mgenre-wiki")
    # model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mgenre-wiki").eval()
    # link_finetuning_dataset(model, tokenizer)

# =======================================  mGenre with HG====================================================

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# tokenizer = AutoTokenizer.from_pretrained("facebook/mgenre-wiki", cache_dir=CACHE_DIR)
# model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mgenre-wiki", cache_dir=CACHE_DIR).eval().to(DEVICE)

# sentences = []
# sentences.append("[START] Einstein [END] era un fisico tedesco.")
# sentences.append("[START] Einstein [END] was a German physicist.")
# sentences.append("how long did it take the [START] twin towers [END] to be built?")
# sentences.append("who sings love you like there's no tomorrow? [START] Chris Young [END]")

# Italian for

# outputs = model.generate(
#     **tokenizer(sentences, return_tensors="pt"),
#     num_beams=2,
#     num_return_sequences=1,
#     # OPTIONAL: use constrained beam search
#     # prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(sent.tolist()),
# )

# res = tokenizer.batch_decode(outputs, skip_special_tokens=True)
# print(res)

# =======================================  mGenre not with HG====================================================


# import pickle
#
# from EntityLinking.genre.genre.fairseq_model import mGENRE
# from EntityLinking.genre.genre.trie import MarisaTrie, Trie
#
# with open("../data/lang_title2wikidataID-normalized_with_redirect.pkl", "rb") as f:
#     lang_title2wikidataID = pickle.load(f)
#
# # memory efficient prefix tree (trie) implemented with `marisa_trie`
# with open("../data/titles_lang_all105_marisa_trie_with_redirect.pkl", "rb") as f:
#     trie = pickle.load(f)
#
# # generate Wikipedia titles and language IDs
# model = mGENRE.from_pretrained("../models/fairseq_multilingual_entity_disambiguation").eval()
#
# model.sample(
#     sentences=["[START] Einstein [END] era un fisico tedesco."],
#     # Italian for "[START] Einstein [END] was a German physicist."
#     prefix_allowed_tokens_fn=lambda batch_id, sent: [
#         e for e in trie.get(sent.tolist()) if e < len(model.task.target_dictionary)
#     ],
#     text_to_id=lambda x: max(lang_title2wikidataID[
#         tuple(reversed(x.split(" >> ")))
#     ], key=lambda y: int(y[1:])),
#     marginalize=True,
# )


# # ================================ Check for number of page views in wikipedia: ==================================
#
# from mwviews.api import PageviewsClient
# p = PageviewsClient(user_agent="<person@organization.org> Selfie, Cat, and Dog analysis")
# res = p.article_views('en.wikipedia', ['Lionel Messi'], granularity="monthly", start="20221201")
# print(res)


# # ===================================== Check details about wikidata entities:   =====================================
#
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


# from wikidata.client import Client
#
# client = Client()
# entity = client.get('Q801', load=True)
# print(entity.description)
# rel = client.get('P361')
# print(rel.description)
# print(entity[rel].description)