import json
import jsonlines
import pandas as pd
import spacy
import torch
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pywikibot

# ===============================      Global Variables:      ===============================

ROOT_PATH = "/cs/labs/oabend/maximifergan/MultilingualKnowledgeTransfer/"
FINETUNING_DATA_PATH = ROOT_PATH + "Data/Datasets/PreprocessDataset.csv"
OUTPUT_PATH = ROOT_PATH + "EntityLinking/FinetuningDatasets/Results/finetuning_entities.json"
CACHE_DIR = ROOT_PATH + "downloaded_models"
START_TOKEN = "[START]"
END_TOKEN = "[END]"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NLP = spacy.load('en_core_web_sm')
for name in NLP.pipe_names:
    if name != 'ner':
        NLP.remove_pipe(name)
# NLP.add_pipe('dbpedia_spotlight')  # pip install spacy-dbpedia-spotlight
SITE = pywikibot.Site("en", "wikipedia")

# ===============================      Global Functions:      ===============================


def link_qa_pair(question, answer, model, tokenizer):

    q_doc = NLP(question)
    q_entities, a_entities = [], []

    # # Using 'dbpedia_spotlight':
    # q_entities = [ent._.dbpedia_raw_result for ent in q_doc.ents]
    # a_entities = [ent._.dbpedia_raw_result for ent in a_doc.ents]

    # Tag question entities:
    for ent in q_doc.ents:
        inf_sent = question[:ent.start_char] + START_TOKEN + " " + \
                   question[ent.start_char:ent.end_char] + " " + END_TOKEN \
                   + question[ent.end_char:]

        outputs = model.generate(**tokenizer(inf_sent, return_tensors="pt"), num_beams=2, num_return_sequences=1)
        entity = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if not entity:
            continue
        entity_name, entity_lang = entity[0].split(" >> ")
        page = pywikibot.Page(SITE, entity_name)
        item = pywikibot.ItemPage.fromPage(page)
        entity_id = item.id
        q_entities.append((entity_name, entity_id))

    # Tag answer entities:
    inf_sent = question + " " + START_TOKEN + " " + answer + " " + END_TOKEN
    outputs = model.generate(**tokenizer(inf_sent, return_tensors="pt"), num_beams=2, num_return_sequences=1)
    entity = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if entity:
        entity_name, entity_lang = entity[0].split(" >> ")
        page = pywikibot.Page(SITE, entity_name)
        item = pywikibot.ItemPage.fromPage(page)
        entity_id = item.id
        a_entities.append((entity_name, entity_id))

    return {'q_entities': q_entities,
            'a_entities': a_entities}


def link_finetuning_dataset(model, tokenizer, input_path=FINETUNING_DATA_PATH, output_path=OUTPUT_PATH):
    df = pd.read_csv(input_path)
    data = df[["Question", "Answer", "Id", "Language"]].to_numpy()
    with open(output_path, 'w') as outfile:
        with jsonlines.Writer(outfile) as writer:
            for i in tqdm(range(data.shape[0])):
                if data[i][3] != "en":
                    continue
                qa_entities = link_qa_pair(data[i][0], data[i][1], model, tokenizer)
                qa_entities["Id"] = data[i][2]
                writer.write(qa_entities)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("facebook/mgenre-wiki", cache_dir=CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mgenre-wiki", cache_dir=CACHE_DIR).eval()
    link_finetuning_dataset(model, tokenizer)

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