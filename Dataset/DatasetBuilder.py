import pandas as pd
import requests
import io
from lodstorage.sparql import SPARQL
from lodstorage.csv import CSV
import lodstorage

# ===============================      Global Variables:      ===============================


# ===============================      Global functions:      ===============================

# Writer, Politician, Sports, Musician, artist

# ====================================      Class:      ====================================


class DatasetBuilder:

    def __init__(self):
        pass


if __name__ == "__main__":
    sparqlQuery = """SELECT ?entitiy ?gender ?langcode ?cityOfBirth ?enLabel ?frLabel ?heLabel ?arLabel ?deLabel ?esLabel ?itLabel ?ruLabel ?jaLabel ?birthYear ?deathYear
    WHERE
    {
    ?entitiy wdt:P31 wd:Q5.  # instance of human
    ?entitiy wdt:P106 wd:Q36180.  # writer
    #   ?entitiy wdt:P106 wd:Q82955.  # politian
    #   ?entitiy wdt:P106 wd:Q82955.  # Sports
    #   ?entitiy wdt:P106 wd:Q82955.  # Musician
    #   ?entitiy wdt:P106 wd:Q82955.  # Artist
    ?entitiy wdt:P569 ?birthDate. # save birth date
    ?entitiy wdt:P570 ?deathDate. # save birth date
    ?entitiy wdt:P21 ?gender.
    ?entitiy wdt:P19 ?cityOfBirth.
    ?cityOfBirth wdt:P17 ?contry.
    ?contry wdt:P37 ?officialLang.
    ?officialLang wdt:P424 ?langcode.
    FILTER(?langcode = "de")
    BIND(YEAR(?birthDate) AS ?birthYear)  # save birth year
    BIND(YEAR(?deathDate) AS ?deathYear)  # save birth year
    FILTER(?birthYear > 1600)
    ?entitiy rdfs:label ?enLabel.
    ?entitiy rdfs:label ?heLabel.
    ?entitiy rdfs:label ?frLabel.
    ?entitiy rdfs:label ?deLabel.
    ?entitiy rdfs:label ?esLabel.
    ?entitiy rdfs:label ?itLabel.
    ?entitiy rdfs:label ?arLabel.
    ?entitiy rdfs:label ?ruLabel.
    ?entitiy rdfs:label ?jaLabel.
    FILTER(LANG(?enLabel) = "en"). # 1000 - 1000
    FILTER(LANG(?heLabel) = "he"). # 23 - 34
    FILTER(LANG(?frLabel) = "fr"). # 558 - 444
    FILTER(LANG(?deLabel) = "de"). # 585 - 565
    FILTER(LANG(?esLabel) = "es"). # 159 - 
    FILTER(LANG(?itLabel) = "it"). # 159 - 
    FILTER(LANG(?arLabel) = "ar"). # 93 - 332
    FILTER(LANG(?ruLabel) = "ru"). # 208 - 
    FILTER(LANG(?jaLabel) = "ja"). # 58 - 83
    }  LIMIT 1000
    """
    sparql = SPARQL("https://query.wikidata.org/sparql")
    qlod = sparql.queryAsListOfDicts(sparqlQuery)
    csv = CSV.toCSV(qlod)
    print(csv)