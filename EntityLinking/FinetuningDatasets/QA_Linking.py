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
# nltk.download('stopwords')
from nltk.corpus import stopwords

# ===============================      Global Variables:      ===============================

# ROOT_PATH = os.path.abspath("")
FINETUNING_DATA_PATH = "/cs/labs/oabend/maximifergan/MultilingualKnowledgeTransfer/Data/Datasets/PreprocessDatasetIntersecet.csv"
OUTPUT_PATH = "/cs/labs/oabend/maximifergan/MultilingualKnowledgeTransfer/EntityLinking/FinetuningDatasets/Results/finetuning_entities.json"
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
    """
    this function links the entities in a given question and answer
    :param question: (String) the question
    :param answer: (String) the answer
    :param model: facebook mGENRE model
    :param tokenizer: facebook mGENRE tokenizer
    :return: A python dictionary that contain the question and answer entities
    """
    # Save the results in:
    q_entities, a_entities = [], []

    # To identifying the nouns in the question
    q_doc = NLP(question)

    # =======================  Tag question entities:  =======================
    for chunk in q_doc.noun_chunks:

        # Filter unwanted nons (Not entities):
        if chunk.text in STOP_WORDS or chunk.text.startswith("how") or chunk.text.startswith("what"):
            continue

        # Build the sentence that will be inference:
        inf_sent = str(question[:chunk.start_char]) + START_TOKEN + " " + \
                   str(question[chunk.start_char:chunk.end_char]) + " " + END_TOKEN \
                   + str(question[chunk.end_char:])

        # inference model
        outputs = model.generate(**tokenizer(inf_sent, return_tensors="pt").to(DEVICE), num_beams=2,
                                 num_return_sequences=1)
        entity = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if not entity:  # didn't found a entity
            continue

        # For the detected entity extract it's wikidata ID:
        tmp = entity[0].split(" >> ")
        if len(tmp) == 2 and tmp[0] != "":
            entity_name, entity_lang = tmp
            try:
                page = pywikibot.Page(SITE, entity_name)
                item = pywikibot.ItemPage.fromPage(page)
                entity_id = item.id
                q_entities.append((entity_name, entity_id))
            except pywikibot.exceptions.NoPageError:
                pass
            except pywikibot.exceptions.InvalidTitleError:
                pass

    # ======================= Tag answer entities: =======================
    inf_sent = START_TOKEN + " " + str(answer) + " " + END_TOKEN  # Build the sentence that will be inference:

    # inference model
    outputs = model.generate(**tokenizer(inf_sent, return_tensors="pt").to(DEVICE), num_beams=2, num_return_sequences=1)
    entity = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    if entity:  # if detected a entity in answer --> extract it's wikidata ID:
        tmp = entity[0].split(" >> ")
        if len(tmp) == 2 and tmp[0] != "":
            entity_name, entity_lang = entity[0].split(" >> ")
            try:
                page = pywikibot.Page(SITE, entity_name)
                item = pywikibot.ItemPage.fromPage(page)
                entity_id = item.id
                a_entities.append((entity_name, entity_id))
            except pywikibot.exceptions.NoPageError:
                pass
            except pywikibot.exceptions.InvalidTitleError:
                pass

    return {'q_entities': q_entities,
            'a_entities': a_entities}


def link_finetuning_dataset(model, tokenizer, input_path=FINETUNING_DATA_PATH, output_path=OUTPUT_PATH):
    """
    This function link all the finetuning datasets QA-pairs with there wikidata entity code.
    :param model: facebook mGENRE model
    :param tokenizer: facebook mGENRE tokenizer
    :param input_path: the finetunning dataset path
    :param output_path: the output path to save the results as a json file
    """
    final_result = pd.read_csv(input_path)
    data = final_result[["Question", "Answer", "Id", "Language", "Dataset"]].to_numpy()

    flag = False  # Flag to limit the range of the tagging --> Set True
    with open(output_path, 'w') as outfile:
        with jsonlines.Writer(outfile) as writer:
            for i in tqdm(range(data.shape[0])):

                # Code for tag only for a start question id and ending in a given end id:
                # if str(data[i][2]) == "1174897936116285345":  # From:
                #     flag = False
                # elif str(data[i][2]) == "4529749965014481176":  # End:
                #     flag = True

                # Only tag english QA and not from Mintaka that is already tagged:
                if flag or data[i][3] != "en" or data[i][4] == "Mintaka":
                    continue

                qa_entities = link_qa_pair(data[i][0], data[i][1], model, tokenizer)  # Tag the current QA-pair
                qa_entities["Id"] = data[i][2]  # Add it's Id

                # Write to json file:
                writer.write(qa_entities)


def main():
    """ Generate all the entity linking """
    tokenizer = AutoTokenizer.from_pretrained("facebook/mgenre-wiki", cache_dir=CACHE_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mgenre-wiki", cache_dir=CACHE_DIR).eval().to(DEVICE)
    link_finetuning_dataset(model, tokenizer)
