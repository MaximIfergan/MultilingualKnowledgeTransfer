import random
import pandas as pd
from mwviews.api import PageviewsClient
from wikidata.client import Client
import pywikibot.data.api as api
import pywikibot
from tqdm.auto import tqdm
import numpy as np
import json

CLIENT = Client()


def get_entity_name(entity_id, lang):
    entity = CLIENT.get(entity_id, load=True)
    page_name = entity.data['labels'][lang]['value']
    return page_name


def get_entity_id(entity_name, lang):
    try:
        site = pywikibot.Site(lang, "wikipedia")
        page = pywikibot.Page(site, entity_name)
        item = pywikibot.ItemPage.fromPage(page)
        entity_id = item.id
        return entity_id
    except pywikibot.exceptions.NoPageError or pywikibot.exceptions.InvalidTitleError:
        return None


def sort_entity_map(entity_map, sample_num=10000):
    res = dict()
    for e in tqdm(entity_map.keys()):
        res[e] = (entity_map[e].shape[0], np.unique(entity_map[e]).shape[0])
    return res


def get_daily_average_page_view(entity_id, lang):
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


def get_number_of_appearance_in_pretraining(training_entities, entity_id):
    page_name = get_entity_name(entity_id, 'en')
    prefix = "_".join([w.capitalize() for w in page_name.split()])
    uri = "http://dbpedia.org/resource/" + prefix
    if uri in training_entities:
        number_of_appearance = training_entities[uri].shape[0]
    else:
        number_of_appearance = None
    return number_of_appearance


def sample_appearance_in_pretraining(entity_map, output_path, sample_num=10000):
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


def build_entity_stats():
    df = pd.DataFrame({"name": [], "source": [],
                       "qa_id": [], "daily_views": [],
                       "wikipedia": [], "lang": []})
    flag = True
    training_entities = np.load("EntityLinking\PretrainingDatasets\wikipedia_entity_map.npz")
    count = 0
    for line in tqdm(
            open("EntityLinking/FinetuningDatasets/Results/finetuning_entities_all.json", 'r', encoding="utf8")):
        qa = json.loads(line)
        count += 1
        if qa["Id"] == "5655493461695504401":
            flag = False
        for entity in qa["q_entities"] + qa["a_entities"]:
            entity_id = entity[1]
            entity_wikipedia = get_number_of_appearance_in_pretraining(training_entities, entity_id)
            if entity_wikipedia is None:
                continue
            entity_name = entity[0]
            entity_daily_views = get_daily_average_page_view(entity_id, 'en')
            qa_id = qa["Id"]
            entity_source = "MKQA" if flag else "NQ"
            df.loc[entity_id] = [entity_name, entity_source, qa_id, entity_daily_views, entity_wikipedia, 'en']
        if count >= 50:
            break
    df.to_csv("first_try.csv")


def main():
    # training_entities = np.load("EntityLinking\PretrainingDatasets\wikipedia_entity_map.npz")
    # training_entities = sort_entity_map(training_entities)
    # print(get_number_of_appearance_in_pretraining(training_entities, 'Q1617977'))
    # sample_appearance_in_pretraining(training_entities, output_path="try.json", sample_num=10000)
    # print(get_number_of_appearance_in_pretraining(training_entities, 'Q22686'))
    build_entity_stats()

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
