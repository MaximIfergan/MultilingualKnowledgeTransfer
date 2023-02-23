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

# ===============================      Global Variables:      ===============================

WIKI_PATH = "EntityLinking/PretrainingDatasets/datasets--nkandpa2--pretraining_entities/snapshots/550b4b11a5ac147bf261ff150a65b98b01469b3f/wikipedia_entity_map.npz"
ROOTS_PATH = "EntityLinking/PretrainingDatasets/datasets--nkandpa2--pretraining_entities/snapshots/550b4b11a5ac147bf261ff150a65b98b01469b3f/roots_entity_map.npz"
C4_PATH = "EntityLinking/PretrainingDatasets/datasets--nkandpa2--pretraining_entities/snapshots/550b4b11a5ac147bf261ff150a65b98b01469b3f/c4_entity_map.npz"
MKQA_TAG_ENTITIES_PATH = "Data/Datasets/MKQA/MKQA_Linked_Entities.json"
NQ_TAG_ENTITIES_PATH = "Data/Datasets/NQ/NQ_Linked_Entities.json"
MINTAKA_TRAIN_DATASET_PATH = "Data/Datasets/Mintaka/mintaka_train.json"
POPQA_DATASET_PATH = "Data/Datasets/POPQA/popQA.tsv"
CLIENT = Client()


# ===============================      Global Functions:      ===============================


def get_entity_name(entity_id, lang):
    """
    :return: The entity name as a string for a given entity id and language code
    """
    entity = CLIENT.get(entity_id, load=True)
    page_name = entity.data['labels'][lang]['value']
    return page_name


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
    site = pywikibot.Site(lang, "wikipedia")
    page = pywikibot.Page(site, page_name)
    req = api.Request(site=site, parameters={'action': 'query',
                                             'titles': page.title(),
                                             'prop': 'pageviews'})
    page_view_stats = req.submit()['query']['pages'][str(page.pageid)]['pageviews']
    total_views = 0
    number_of_days = 0
    for key in page_view_stats:
        if page_view_stats[key] is None:
            continue
        total_views += page_view_stats[key]
        number_of_days += 1
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
            entity_daily_views = get_daily_average_page_view(entity_id, 'en')
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
            entity_daily_views = get_daily_average_page_view(entity_id, 'en')
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

            # Limit the number of entities to add:
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
                entity_daily_views = get_daily_average_page_view(entity_id, 'en')
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
        entity_daily_views = get_daily_average_page_view(entity_id, 'en')
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


def create_df_for_stats():
    """ this function builds the entities dataframe for stats analysis """
    # training_entities = np.load(WIKI_PATH)
    training_entities = np.load("EntityLinking/PretrainingDatasets/wikipedia_entity_map.npz")
    df = pd.DataFrame({"name": [], "source": [],
                       "qa_id": [], "daily_views": [],
                       "wikipedia": [], "lang": [],
                       "dbpedia_uri": []})
    df = add_MKQA_entities(df, training_entities, num_of_entities=5)
    df = add_NQ_entities(df, training_entities, num_of_entities=5)
    df = add_Mintaka_entities(df, training_entities, num_of_entities=5)
    df = add_PopQA_entities(df, training_entities, num_of_entities=5)
    # df = add_pretraining_dataset_appearance(df)
    return df


def main():
    training_entities = np.load("EntityLinking/PretrainingDatasets/wikipedia_entity_map.npz")
    # df = pd.read_csv("entities_stats_final.csv")
    # df = df[df["source"] == "PopQA"]
    # print(df["daily_views"].corr(df["c4"]))
    # df.hist(bins=3)
    # plt.show()


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
            for line in open("EntityLinking/FinetuningDatasets/Results/finetuning_entities_all.json", 'r',
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
