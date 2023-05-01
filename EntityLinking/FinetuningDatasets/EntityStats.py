import random
import pandas as pd
# from mwviews.api import PageviewsClient
from jsonlines import jsonlines
from wikidata.client import Client
import pywikibot.data.api as api
import pywikibot
from tqdm.auto import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
import Data.DataPreprocessing as DataPreprocessing
import sys
import pickle
import re

# ===============================      Global Variables:      ===============================

WIKI_PATH = "EntityLinking/PretrainingDatasets/wikipedia_entity_map.npz"
ROOTS_PATH = "EntityLinking/PretrainingDatasets/roots_entity_map.npz"
C4_PATH = "EntityLinking/PretrainingDatasets/c4_entity_map.npz"
MKQA_TAG_ENTITIES_PATH = "Data/Datasets/MKQA/MKQA_Linked_Entities.json"
NQ_TAG_ENTITIES_PATH = "Data/Datasets/NQ/NQ_Linked_Entities.json"
MINTAKA_TRAIN_DATASET_PATH = "Data/Datasets/Mintaka/mintaka_train.json"
POPQA_DATASET_PATH = "Data/Datasets/POPQA/popQA.tsv"
MKQA_ENTITIES_PATH = "Data/Datasets/MKQA/MKQA_Linked_Entities.json"
MKQA_ENTITIES_TO_PV = "EntityLinking/FinetuningDatasets/Results/MKQA_entities_to_pv.pkl"
MINTAKA_ENTITIES_TO_PV = "EntityLinking/FinetuningDatasets/Results/Mintaka_entities_to_pv.pkl"
MINTAKA_DEV_DATASET_PATH = "Data/Datasets/Mintaka/mintaka_dev.json"
MINTAKA_TEST_DATASET_PATH = "Data/Datasets/Mintaka/mintaka_test.json"
CLIENT = Client()


# ===============================      Global Functions:      ===============================


def get_entity_name(entity_id, lang):
    """
    :return: The entity name as a string for a given entity id and language code
    """
    try:
        entity = CLIENT.get(entity_id, load=True)
    except Exception as e:
        sys.stderr.write("\n Error:" + str(e) + "\n")
        return -1
    if lang in entity.data['labels']:
        return entity.data['labels'][lang]['value']
    else:
        return -1


def get_entity_id(entity_name, lang):
    """
    :return: given a entity name and language code returns the entity wikidata code if exist else None
    """
    try:
        site = pywikibot.Site(lang, "wikipedia")
        page = pywikibot.Page(site, entity_name)
        item = pywikibot.ItemPage.fromPage(page)
        entity_id = item.id
        return entity_id
    except pywikibot.exceptions.NoPageError or pywikibot.exceptions.InvalidTitleError:
        return None


def get_daily_average_page_view(entity_id, lang):
    """ this function for a given entity id and language returns the number of the average daily views of it's
        wikipedia page in the given language """
    page_name = get_entity_name(entity_id, lang)
    if page_name == -1:
        return -1, -1
    site = pywikibot.Site(lang, "wikipedia")
    page = pywikibot.Page(site, page_name)
    try:
        req = api.Request(site=site, parameters={'action': 'query',
                                                 'titles': page.title(),
                                                 'prop': 'pageviews'})
        page_view_stats = req.submit()['query']['pages'][str(page.pageid)]['pageviews']
    except (KeyError, pywikibot.exceptions.InvalidTitleError) as e:
        return -1, -1
    total_views = 0
    number_of_days = 0
    for key in page_view_stats:
        if page_view_stats[key] is None:
            continue
        total_views += page_view_stats[key]
        number_of_days += 1
    if number_of_days == 0:
        return -1, -1
    return int(total_views / number_of_days), page_name


def get_daily_average_pv(page, site):
    """ this function for a given entity id and language returns the number of the average daily views of it's
        wikipedia page in the given language """
    try:
        req = api.Request(site=site, parameters={'action': 'query',
                                                 'titles': page.title(),
                                                 'prop': 'pageviews'})
        page_view_stats = req.submit()['query']['pages'][str(page.pageid)]['pageviews']
    except (KeyError, pywikibot.exceptions.InvalidTitleError) as e:
        return -1
    total_views = 0
    number_of_days = 0
    for key in page_view_stats:
        if page_view_stats[key] is None:
            continue
        total_views += page_view_stats[key]
        number_of_days += 1
    if number_of_days == 0:
        return -1
    return int(total_views / number_of_days)


def get_number_of_appearance_in_pretraining(training_entities, entity_name, type="name"):
    """
    this function given a entity id returns the number appearance of the entity in the the pretraining dataset.
    :param type:
    :param training_entities: the dictionary that maps entities to the number of appearance
    :param entity_name: the entity id
    :return: it's key and number of appearance and None if didn't found entity
    """
    if type == "id":
        entity_name = get_entity_name(entity_name, 'en')

    # Build the key for the dictionary:
    prefix = "_".join([w.capitalize() for w in entity_name.split()])
    uri = "http://dbpedia.org/resource/" + prefix

    # Extract form the dictionary
    if uri in training_entities:
        number_of_appearance = training_entities[uri].shape[0]
    else:
        number_of_appearance = None
    return uri, number_of_appearance


# ======== Build Entity stats for views and appearance in the pretraining datasets  =========


def add_MKQA_entities(df, training_entities, num_of_entities=float('inf'), mkqa_entities_path=MKQA_TAG_ENTITIES_PATH):
    """
    this function adds MKQA entities to a given dataframe with their wikipedia page daily views and pretraining dataset
    appearance for stats analysis.
    :param df: the given dataframe
    :param training_entities: a pretraining entities to match the keys
    :param num_of_entities: the number of entities to add from MKQA
    :param mkqa_entities_path: the path to the tag QA-pairs
    :return: the df with the added entities
    """
    count = 0  # To limit the number of entities added

    for line in open(mkqa_entities_path, 'r', encoding="utf8"):
        qa = json.loads(line)

        # Limit the number of entities to add:
        if count >= num_of_entities:
            continue

        # Add all the QA-pairs entities:
        for entity in qa["q_entities"] + qa["a_entities"]:
            entity_id = entity[1]
            entity_name = entity[0]
            key, num_of_appearance_in_pretraining = get_number_of_appearance_in_pretraining(training_entities,
                                                                                            entity_name)
            if num_of_appearance_in_pretraining is None:
                continue
            entity_daily_views, _ = get_daily_average_page_view(entity_id, 'en')
            qa_id = qa["Id"]
            df.loc[entity_id] = [entity_name, "MKQA", qa_id, entity_daily_views,
                                 num_of_appearance_in_pretraining, 'en', key]
            count += 1
    return df


def add_NQ_entities(df, training_entities, num_of_entities=float('inf'), nq_entities_path=NQ_TAG_ENTITIES_PATH):
    """
    this function adds NQ entities to a given dataframe with their wikipedia page daily views and pretraining dataset
    appearance for stats analysis.
    :param df: the given dataframe
    :param training_entities: a pretraining entities to match the keys
    :param num_of_entities: the number of entities to add from NQ
    :param nq_entities_path: the path to the tag QA-pairs
    :return: the df with the added entities
    """

    count = 0  # To limit the number of entities added

    for line in open(nq_entities_path, 'r', encoding="utf8"):
        qa = json.loads(line)

        # Limit the number of entities to add:
        if count >= num_of_entities:
            continue

        # Add all the QA-pair entities:
        for entity in qa["q_entities"] + qa["a_entities"]:
            entity_id = entity[1]
            entity_name = entity[0]
            key, num_of_appearance_in_pretraining = get_number_of_appearance_in_pretraining(training_entities,
                                                                                            entity_name)
            if num_of_appearance_in_pretraining is None:
                continue
            entity_daily_views, _ = get_daily_average_page_view(entity_id, 'en')
            qa_id = qa["Id"]
            df.loc[entity_id] = [entity_name, "NQ", qa_id, entity_daily_views,
                                 num_of_appearance_in_pretraining, 'en', key]
            count += 1
    return df


def add_Mintaka_entities(df, training_entities, num_of_entities=float('inf')):
    """
    this function adds Mintaka entities to a given dataframe with their wikipedia page daily views and pretraining
    dataset appearance for stats analysis.
    :param df: the given dataframe
    :param training_entities: a pretraining entities to match the keys
    :param num_of_entities: the number of entities to add from Mintaka
    :return: the df with the added entities
    """

    count = 0  # To limit the number of entities added

    with open(MINTAKA_TRAIN_DATASET_PATH, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        for qa in data:

            # Limit the number of entities to add :
            if count >= num_of_entities:
                break

            # Filter (Same as the one used to build finetuning dataset) :
            if qa['answer']['answer'] is not None and len(qa['answer']['answer']) > 1:
                continue

            # Add all the QA-pair entities:
            for entity in qa["questionEntity"]:
                if entity["entityType"] != "entity":
                    continue
                entity_id = entity["name"]
                key, num_of_appearance_in_pretraining = get_number_of_appearance_in_pretraining(training_entities,
                                                                                                entity_id, type="id")
                if num_of_appearance_in_pretraining is None:
                    continue
                entity_name = entity["label"]
                entity_daily_views, _ = get_daily_average_page_view(entity_id, 'en')
                qa_id = qa["id"]
                df.loc[entity_id] = [entity_name, "Mintaka", qa_id, entity_daily_views,
                                     num_of_appearance_in_pretraining, 'en', key]
                count += 1
    return df


def add_PopQA_entities(df, training_entities, num_of_entities=float('inf')):
    """
    this function adds PopQA entities to a given dataframe with their wikipedia page daily views and pretraining dataset
    appearance for stats analysis.
    :param df: the given dataframe
    :param training_entities: a pretraining entities to match the keys
    :param num_of_entities: the number of entities to add from PopQA
    :return: the df with the added entities
    """

    count = 0  # To limit the number of entities added

    data = pd.read_csv(POPQA_DATASET_PATH, sep='\t')
    s_data = data.sample(frac=1).reset_index(drop=True)

    for index, row in s_data.iterrows():

        # Limit the number of entities to add:
        if count >= num_of_entities:
            break

        entity_id = row["s_uri"].split("/")[-1]
        entity_name = row["s_wiki_title"]
        key, num_of_appearance_in_pretraining = get_number_of_appearance_in_pretraining(training_entities, entity_id,
                                                                                        type="id")
        if num_of_appearance_in_pretraining is None:
            continue
        entity_daily_views, _ = get_daily_average_page_view(entity_id, 'en')
        qa_id = row["id"]
        df.loc[entity_id] = [entity_name, "PopQA", qa_id, entity_daily_views,
                             num_of_appearance_in_pretraining, 'en', key]
        count += 1
    return df


def add_pretraining_dataset_appearance(df):
    """ this functions add the pretraining dataset entities appearance to the given dataframe """
    c4 = np.load(C4_PATH)
    roots = np.load(ROOTS_PATH)
    df["c4"] = -1
    df["roots"] = -1
    for index, row in tqdm(df.iterrows()):
        key = row["dbpedia_uri"]
        try:
            df.at[index, "c4"] = c4[key].shape[0]
        except KeyError:
            print(f"c4: didn't found {key}")
        try:
            df.at[index, "roots"] = roots[key].shape[0]
        except KeyError:
            print(f"roots: didn't found {key}")
    return df


def add_entities_from_query(df, training_entities, file_path, num_of_entities=float('inf')):
    count = 0
    new_entities_df = pd.read_csv(file_path)
    source = file_path.split("/")[-1]
    for index, row in new_entities_df.iterrows():
        if count > num_of_entities:
            break
        entity_id = row["item"].split("/")[-1]
        entity_name = row["itemLabel"]
        key, num_of_appearance_in_pretraining = get_number_of_appearance_in_pretraining(training_entities, entity_name)
        if num_of_appearance_in_pretraining is None:
            continue
        entity_daily_views, _ = get_daily_average_page_view(entity_id, 'en')
        df.loc[entity_id] = [entity_name, source, index, entity_daily_views,
                             num_of_appearance_in_pretraining, 'en', key]
        count += 1
    return df


def create_df_for_stats():
    """ this function builds the entities dataframe for stats analysis """
    training_entities = np.load(WIKI_PATH)
    # training_entities = np.load("EntityLinking/PretrainingDatasets/wikipedia_entity_map.npz")
    df = pd.DataFrame({"name": [], "source": [],
                       "qa_id": [], "daily_views": [],
                       "wikipedia": [], "lang": [],
                       "dbpedia_uri": []})
    df = add_MKQA_entities(df, training_entities, num_of_entities=300)
    df = add_NQ_entities(df, training_entities, num_of_entities=300)
    df = add_Mintaka_entities(df, training_entities, num_of_entities=300)
    df = add_PopQA_entities(df, training_entities, num_of_entities=500)
    add_entities_from_query(df, training_entities,
                            "EntityLinking/FinetuningDatasets/QueriesResultsForStats/query_cities.csv")
    add_entities_from_query(df, training_entities,
                            "EntityLinking/FinetuningDatasets/QueriesResultsForStats/query_companies.csv")
    add_entities_from_query(df, training_entities,
                            "EntityLinking/FinetuningDatasets/QueriesResultsForStats/query_countries.csv")
    add_entities_from_query(df, training_entities,
                            "EntityLinking/FinetuningDatasets/QueriesResultsForStats/query_religions.csv")
    df = add_pretraining_dataset_appearance(df)
    return df


def plot_corr_group_by_source(dataset="c4"):
    corr = []
    df = pd.read_csv("EntityLinking/FinetuningDatasets/results_for_stats.csv")
    sources = list(df["source"].unique())
    for source in sources:
        corr.append(df[df["source"] == source]["daily_views"].corr(df[dataset]))
    for i in range(len(sources)):
        if sources[i].startswith("query"):
            sources[i] = sources[i][6:-4]
    sources.append("All")
    corr.append(df["daily_views"].corr(df[dataset]))
    plt.title(f"Correlation Between page views and {dataset}")
    plt.bar(sources, corr)
    plt.xticks(rotation=45, size="small")
    plt.xlabel("Entities sources")
    plt.ylabel("corr")
    for i in range(len(sources)):
        plt.text(i, round(corr[i], 2), round(corr[i], 2), ha='center')
    plt.show()


def add_to_PopQA_page_views(path=POPQA_DATASET_PATH):
    sites_dict = dict()  # Save all the Wikipedia sites in use:
    cash_memory = dict()  # Cash the memory of entities that were in use:
    for lang in DataPreprocessing.FINETUNING_LANGS_INTERSEC:
        sites_dict[lang] = pywikibot.Site(lang, "wikipedia")
        cash_memory[lang] = dict()

        # Start form scratch:
        # data[f"{lang}_s_pv"] = -1
        # data[f"{lang}_o_pv"] = -1
        # data[f"{lang}_s_label"] = -1
        # data[f"{lang}_o_label"] = -1

    # Load the data:
    # data = pd.read_csv(path, sep='\t')   # Start form scratch
    data = pd.read_csv(path, delimiter=",")  # Start after some extractions

    count = 0  # For debuging
    for index, row in data.iterrows():

        # Start from an index
        if count <= 7299:
            count += 1
            continue

        # backup after some iterations
        if count % 200 == 0:
            data.to_csv("backup.csv")
            print(f"{count} saved")

        s_id = row["s_uri"].split("/")[-1]  # subject id
        o_id = row["o_uri"].split("/")[-1]  # object id

        s_wikidata_entity = CLIENT.get(s_id, load=True)  # wikidata subject
        o_wikidata_entity = CLIENT.get(o_id, load=True)  # wikidata object

        for lang in DataPreprocessing.FINETUNING_LANGS_INTERSEC:

            # Extract labels:
            s_label = s_wikidata_entity.label[lang] if lang in s_wikidata_entity.label else -1
            o_label = o_wikidata_entity.label[lang] if lang in o_wikidata_entity.label else -1
            s_pv = -1
            o_pv = -1

            # Get pv for subject:
            if s_id in cash_memory[lang]:
                s_pv, s_label = cash_memory[lang][s_id]
            elif s_label != -1:
                try:
                    s_page = pywikibot.Page(sites_dict[lang], s_label)
                    s_pv = get_daily_average_pv(s_page, sites_dict[lang])
                except Exception as e:
                    sys.stderr.write(
                        "\nError:" + str(e) + f"entity_name: {s_label} lang: {lang} entity_id: {s_id} " + "\n")
                    continue
                cash_memory[lang][s_id] = (s_pv, s_label)

            # Get pv for object:
            if o_id in cash_memory[lang]:
                o_pv, o_label = cash_memory[lang][o_id]
            elif o_label != -1:
                try:
                    o_page = pywikibot.Page(sites_dict[lang], o_label)
                    o_pv = get_daily_average_pv(o_page, sites_dict[lang])
                except Exception as e:
                    sys.stderr.write(
                        "\nError:" + str(e) + f"entity_name: {o_label} lang: {lang} entity_id: {o_id} " + "\n")
                    continue
                cash_memory[lang][o_id] = (o_pv, o_label)

            # Save results
            data.at[index, f"{lang}_s_pv"] = s_pv
            data.at[index, f"{lang}_o_pv"] = o_pv
            data.at[index, f"{lang}_s_label"] = s_label
            data.at[index, f"{lang}_o_label"] = o_label

        count += 1
    data.to_csv("PopQA_pv_stats.csv")


def save_mkqa_entities_page_views(entities_path=MKQA_ENTITIES_PATH, output_path=MKQA_ENTITIES_TO_PV):
    # result_dict = dict()
    with open("EntityLinking/FinetuningDatasets/Results/entities_to_pv.pkl", "rb") as fp:
        result_dict = pickle.load(fp)
    sites_dict = dict()
    count = 0
    for lang in DataPreprocessing.FINETUNING_LANGS_INTERSEC:
        sites_dict[lang] = pywikibot.Site(lang, "wikipedia")
        # result_dict[lang] = dict()
    for line in open(entities_path, 'r', encoding='utf8'):
        qa_entities = json.loads(line)
        for entity in qa_entities["q_entities"] + qa_entities["a_entities"]:
            # if count >= 200:
            #     break
            entity_id = entity[1]
            if entity_id in result_dict["en"]:
                continue
            wikidata_entity = CLIENT.get(entity_id, load=True)
            for lang in DataPreprocessing.FINETUNING_LANGS_INTERSEC:
                if count % 5000 == 0:
                    with open(output_path, "wb") as fp:
                        pickle.dump(result_dict, fp)
                        sys.stderr.write(f"\n=============== Backup {count} ===============\n")
                if lang not in wikidata_entity.label:
                    continue
                entity_name = wikidata_entity.label[lang]
                try:
                    page = pywikibot.Page(sites_dict[lang], entity_name)
                    entity_pv = get_daily_average_pv(page, sites_dict[lang])
                except Exception as e:
                    sys.stderr.write(
                        "\nError:" + str(e) + f"entity_name: {entity_name} lang: {lang} entity_id: {entity_id} " + "\n")
                    continue
                result_dict[lang][entity_id] = (entity_name, entity_pv)
                count += 1
    with open(output_path, "wb") as fp:
        pickle.dump(result_dict, fp)


def save_mintaka_entities_page_views(entities_path=MINTAKA_TRAIN_DATASET_PATH, output_path=MINTAKA_ENTITIES_TO_PV):
    # result_dict = dict()
    with open(MINTAKA_ENTITIES_TO_PV, "rb") as fp:
        result_dict = pickle.load(fp)
    sites_dict = dict()
    count = 0
    for lang in DataPreprocessing.FINETUNING_LANGS_INTERSEC:
        sites_dict[lang] = pywikibot.Site(lang, "wikipedia")
        # result_dict[lang] = dict()
    with open(entities_path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        for qa in data:

            # Filter (Same as the one used to build finetuning dataset) :
            if qa['answer']['answer'] is not None and len(qa['answer']['answer']) > 1:
                continue

            # # Add all the QA-pair entities and their pv:
            # for entity in qa["questionEntity"]:
            #     # if count >= 20:
            #     #     break
            #     if entity["entityType"] != "entity" or entity["name"] is None:
            #         continue
            #     entity_id = entity["name"]
            #     if entity_id in result_dict["en"]:
            #         continue
            #     wikidata_entity = CLIENT.get(entity_id, load=True)
            #     for lang in DataPreprocessing.FINETUNING_LANGS:
            #         if count % 1000 == 0:
            #             with open(output_path, "wb") as fp:
            #                 pickle.dump(result_dict, fp)
            #                 sys.stderr.write(f"\n=============== Backup {count} ===============\n")
            #         if wikidata_entity.label is None or lang not in wikidata_entity.label:
            #             continue
            #         entity_name = wikidata_entity.label[lang]
            #         try:
            #             page = pywikibot.Page(sites_dict[lang], entity_name)
            #             entity_pv = get_daily_average_pv(page, sites_dict[lang])
            #         except Exception as e:
            #             sys.stderr.write("\nError:" + str(
            #                 e) + f"entity_name: {entity_name} lang: {lang} entity_id: {entity_id} " + "\n")
            #             continue
            #         result_dict[lang][entity_id] = (entity_name, entity_pv)
            #         count += 1

            if qa['answer']['answerType'] == "entity" and qa['answer']['answer'] is not None:
                answer_entity = qa['answer']['answer'][0]
                if re.match("^Q[1-9]+", answer_entity["name"]):
                    entity_id = answer_entity["name"]
                    if entity_id in result_dict["en"]:
                        continue
                    try:
                        wikidata_entity = CLIENT.get(entity_id, load=True)
                    except Exception as e:
                        sys.stderr.write("\nError:" + str(e) + f"entity_id: {entity_id} " + "\n")
                        continue
                    for lang in DataPreprocessing.FINETUNING_LANGS_INTERSEC:
                        if count % 1000 == 0:
                            with open(output_path, "wb") as fp:
                                pickle.dump(result_dict, fp)
                                sys.stderr.write(f"\n=============== Backup {count} ===============\n")
                        if wikidata_entity.label is None or lang not in wikidata_entity.label:
                            continue
                        entity_name = wikidata_entity.label[lang]
                        try:
                            page = pywikibot.Page(sites_dict[lang], entity_name)
                            entity_pv = get_daily_average_pv(page, sites_dict[lang])
                        except Exception as e:
                            sys.stderr.write("\nError:" + str(
                                e) + f"entity_name: {entity_name} lang: {lang} entity_id: {entity_id} " + "\n")
                            continue
                        result_dict[lang][entity_id] = (entity_name, entity_pv)
                        count += 1

    with open(output_path, "wb") as fp:
        pickle.dump(result_dict, fp)


def main():
    save_mintaka_entities_page_views(entities_path=MINTAKA_TRAIN_DATASET_PATH)
    sys.stderr.write("\n\n\nEnded Train start Test!\n\n\n")
    save_mintaka_entities_page_views(entities_path=MINTAKA_DEV_DATASET_PATH)
    sys.stderr.write("\n\n\nEnded Dev start Test!\n\n\n")
    save_mintaka_entities_page_views(entities_path=MINTAKA_TEST_DATASET_PATH)
    sys.stderr.write("\n\n\nEnded Test start Test!\n\n\n")
    # TODO: clean up form old saving
    # add_to_PopQA_page_views("backup.csv")
    # get_daily_average_page_view("Q2", "fr")
    # save_entities_page_views()


# # =============== Check for number of page views in wikipedia with page view: ======================

# from mwviews.api import PageviewsClient
# p = PageviewsClient(user_agent="<person@organization.org> Selfie, Cat, and Dog analysis")
# res = p.article_views('en.wikipedia', ['Lionel Messi'], granularity="monthly", start="20221201")
# print(res)

# # ================      Check details about wikidata entities:     =====================

# import pywikibot

# site = pywikibot.Site("fr", "wikipedia")
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


# ==============================      Function To delete      ==============================


def separate_dataset_entities_linking():
    df = pd.read_csv("Data/Datasets/PreprocessDataset.csv")
    NQentitiesPath = "NQ_Linked_Entities.json"
    MKQAentitiesPath = "MKQA_Linked_Entities.json"
    with open(NQentitiesPath, 'w', encoding="utf8") as NQoutput, open(MKQAentitiesPath, 'w',
                                                                      encoding="utf8") as MKQAoutput:
        with jsonlines.Writer(NQoutput) as NQwriter, jsonlines.Writer(MKQAoutput) as MKQAwriter:
            index = 0
            for line in open("EntityLinking/FinetuningDatasets/Results/finetuning_entities.json", 'r',
                             encoding="utf8"):
                qa = json.loads(line)
                while df.iloc[index]["Dataset"] == "Mintaka":
                    index += 1
                if qa["Id"] != df.iloc[index]["Id"]:
                    raise Exception(f"index {index} Id's are different. en: {qa['Id']}, dataset {df.iloc[index]['Id']}")
                if df.iloc[index]["Dataset"] == "NQ":
                    NQwriter.write(qa)
                    index += 1
                elif df.iloc[index]["Dataset"] == "MKQA":
                    MKQAwriter.write(qa)
                    index += 6
                else:
                    raise Exception(f"different dataset {df.iloc[index]['Dataset']}")
    print(index)


def sample_appearance_in_pretraining(entity_map, output_path, sample_num=10000):
    # TODO: Test this function and check if she is necessary
    res_lst = []
    count = 0
    for e in tqdm(entity_map.keys()):
        res_lst.append((e, entity_map[e].shape[0]))
        count += 1
        if count >= sample_num * 2:
            break
    random.shuffle(res_lst)
    res_dic = dict()
    for key, value in res_lst[:sample_num]:
        res_dic[key] = value
    with open(output_path, "w") as outfile:
        json.dump(res_dic, outfile)
