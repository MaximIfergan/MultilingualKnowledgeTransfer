import json

import pandas as pd
import requests
import io
import lodstorage  # pip install pyLodStorage
from lodstorage.sparql import SPARQL
from lodstorage.csv import CSV
import random

# ===============================      Global Variables:      ===============================

PROMPT_TEMPLATES = {"birth_year":
                        {"en": {"male": ["{} was born in the year ", "The birth year of {} is "],
                                "female": ["{} was born in the year ", "The birth year of {} is "]},
                         "fr": {"male": ["{} est né en l'an ", "L'année de naissance de {} est "],
                                "female": ["{} est née en l'an ", "L'année de naissance de {} est "]},
                         "ru": {"male": ["{} родился в году ", "Год рождения {} он "],
                                "female": ["{} родилась в году ", "Год рождения {} он "]}},
                    "birth_city":
                        {"en": {"male": ["{} was born in the city ", "The birth city of {} was "],
                                "female": ["{} was born in the city ", "The birth city of {} was "]},
                         "fr": {"male": ["{} est né dans une ville nommée  ", "La ville natale de {} était "],
                                "female": ["{} est née dans une ville nommée  ", "La ville natale de {} était "]},
                         "ru": {"male": ["{} родился в городе ", "Город рождения {} он "],
                                "female": ["{} родилась в городе ", "Город рождения {} он "]}}
                    }

TEMPLATE = {
    "prompt": "",
    "subject": "",
    "target": "",
    "queries": []
}

# ===============================      Global functions:      ===============================

def query_online(sparqlQuery):
    sparql = SPARQL("https://query.wikidata.org/sparql")
    qlod = sparql.queryAsListOfDicts(sparqlQuery)
    csv = CSV.toCSV(qlod)
    return pd.read_csv(io.StringIO(csv))


# ====================================      Class:      ====================================


class DatasetBuilder:

    def __init__(self, data):
        self.raw_data = data
        self.data = data.copy()

    def create_dataset(self):
        dataset = []
        for i in range(len(self.data)):

            # Init samples:
            year_sample = {
                'prompt': PROMPT_TEMPLATES['birth_year']['en'][self.data.loc[i, 'genderLabel']][0],
                'subject': self.data.loc[i, 'o_en'],
                'target': str(self.data.loc[i, 'n_year']),
                'queries': []
            }
            city_sample = {
                'prompt': PROMPT_TEMPLATES['birth_city']['en'][self.data.loc[i, 'genderLabel']][0],
                'subject': self.data.loc[i, 'o_en'],
                'target': self.data.loc[i, 'nc_en'],
                'queries': []
            }

            # Add queries:
            year_sample['queries'].append(
                PROMPT_TEMPLATES['birth_year']['en'][self.data.loc[i, 'genderLabel']][0].format(self.data.loc[i, 'o_en']))
            year_sample['queries'].append(
                PROMPT_TEMPLATES['birth_year']['en'][self.data.loc[i, 'genderLabel']][1].format(self.data.loc[i, 'o_en']))
            year_sample['queries'].append(
                PROMPT_TEMPLATES['birth_year']['fr'][self.data.loc[i, 'genderLabel']][0].format(self.data.loc[i, 'o_fr']))
            year_sample['queries'].append(
                PROMPT_TEMPLATES['birth_year']['fr'][self.data.loc[i, 'genderLabel']][1].format(self.data.loc[i, 'o_fr']))
            year_sample['queries'].append(
                PROMPT_TEMPLATES['birth_year']['ru'][self.data.loc[i, 'genderLabel']][0].format(self.data.loc[i, 'o_ru']))
            year_sample['queries'].append(
                PROMPT_TEMPLATES['birth_year']['ru'][self.data.loc[i, 'genderLabel']][0].format(self.data.loc[i, 'o_ru']))

            city_sample['queries'].append(
                PROMPT_TEMPLATES['birth_city']['en'][self.data.loc[i, 'genderLabel']][0].format(self.data.loc[i, 'o_en']))
            city_sample['queries'].append(
                PROMPT_TEMPLATES['birth_city']['en'][self.data.loc[i, 'genderLabel']][1].format(self.data.loc[i, 'o_en']))
            city_sample['queries'].append(
                PROMPT_TEMPLATES['birth_city']['fr'][self.data.loc[i, 'genderLabel']][0].format(self.data.loc[i, 'o_fr']))
            city_sample['queries'].append(
                PROMPT_TEMPLATES['birth_city']['fr'][self.data.loc[i, 'genderLabel']][1].format(self.data.loc[i, 'o_fr']))
            city_sample['queries'].append(
                PROMPT_TEMPLATES['birth_city']['ru'][self.data.loc[i, 'genderLabel']][0].format(self.data.loc[i, 'o_ru']))
            city_sample['queries'].append(
                PROMPT_TEMPLATES['birth_city']['ru'][self.data.loc[i, 'genderLabel']][0].format(self.data.loc[i, 'o_ru']))

            # Add to dataset
            dataset.append(year_sample)
            dataset.append(city_sample)

        self.dataset = dataset

    def add_target_info(self):
        n_years = []
        nc_en = []
        nc_fr = []
        nc_ru = []
        for i in range(len(self.data)):
            diff = random.randint(1, 10)
            if random.randint(0, 1):
                n_years.append(self.data.loc[i, 'birthYear'] + diff)
            else:
                n_years.append(self.data.loc[i, 'birthYear'] - diff)
            i_n_city = i
            while self.data.loc[i, 'c_en'] == self.data.loc[i_n_city, 'c_en']:
                i_n_city = random.randint(0, len(self.data) - 1)
            nc_en.append(self.data.loc[i_n_city, 'c_en'])
            nc_fr.append(self.data.loc[i_n_city, 'c_fr'])
            nc_ru.append(self.data.loc[i_n_city, 'c_ru'])
        self.data['n_year'] = n_years
        self.data['nc_en'] = nc_en
        self.data['nc_fr'] = nc_fr
        self.data['nc_ru'] = nc_ru

    def save_as_json(self, path):
        # chunks = [self.dataset[i:i + 50] for i in range(0, len(self.dataset), 50)]
        # for i, chunk in enumerate(chunks):
        #     with open(f"QueiriesData/Nobel/nobel_dataset_{i}.json", "w", encoding='utf8') as fp:
        #         json.dump(chunk, fp, ensure_ascii=False)
        with open(path, "w", encoding='utf8') as fp:
            json.dump(self.dataset, fp, ensure_ascii=False)


if __name__ == "__main__":
    db = DatasetBuilder(pd.read_csv("QueiriesData/F_people.csv"))
    db.add_target_info()
    db.create_dataset()
    db.save_as_json("F_people.json")
