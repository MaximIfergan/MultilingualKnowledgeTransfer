import json
import jsonlines
import pandas as pd
import spacy
import numpy as np
from tqdm.auto import tqdm

# ===============================      Global Variables:      ===============================

FINETUNING_DATA_PATH = "Data/Datasets/PreprocessDataset.csv"
OUTPUT_PATH = "EntityLinking/FinetuningDatasets/Results/finetuning_entities.json"
# NLP = spacy.blank('en_core_web_sm')
# for name in NLP.pipe_names:
#     if name != 'ner':
#         NLP.remove_pipe(name)
# NLP.add_pipe('dbpedia_spotlight')  # pip install spacy-dbpedia-spotlight

# ===============================      Global Functions:      ===============================


def link_qa_pair(question, answer):
    q_doc = NLP(question)
    a_doc = NLP(answer)
    # Using 'dbpedia_spotlight':
    q_entities = [ent._.dbpedia_raw_result for ent in q_doc.ents]
    a_entities = [ent._.dbpedia_raw_result for ent in a_doc.ents]
    return {'q_entities': q_entities,
            'a_entities': a_entities}


def link_finetuning_dataset(input_path=FINETUNING_DATA_PATH, output_path=OUTPUT_PATH):
    df = pd.read_csv(input_path)
    data = df[["Question", "Answer", "Id", "Language"]].to_numpy()
    with open(output_path, 'w') as outfile:
        with jsonlines.Writer(outfile) as writer:
            for i in tqdm(range(data.shape[0])):
                if data[i][3] != "en":
                    continue
                qa_entities = link_qa_pair(data[i][0], data[i][1])
                qa_entities["Id"] = data[i][2]
                writer.write(qa_entities)


# ===========================================================================================


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