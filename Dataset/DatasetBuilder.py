import pandas as pd
import requests
import io
import lodstorage  # pip install pyLodStorage
from lodstorage.sparql import SPARQL
from lodstorage.csv import CSV


# ===============================      Global Variables:      ===============================


# ===============================      Global functions:      ===============================

# Writer, Politician, Sports, Musician, artist

# ====================================      Class:      ====================================


class DatasetBuilder:

    def __init__(self):
        pass

def create_entities_query():
    pass


if __name__ == "__main__":
    sparqlQuery = """SELECT ?entitiy ?gender ?langcode ?cityOfBirth ?occupation ?o_en
    ?o_he
    ?o_fr
    ?o_ar
    # ?o_de
    ?o_es
    ?o_ru
    ?o_ja
    ?birthYear
    ?deathYear
    WHERE
    {
      ?entitiy wdt:P31 wd:Q5.  # instance of human
      ?entitiy wdt:P106 ?occupation.  # occupation
      VALUES ?occupation {wd:Q33999 wd:Q639669 wd:Q82955 wd:Q483501 wd:Q1930187 wd:Q901 wd:Q36180 wd:Q177220 wd:Q2066131 wd:Q753110}
    #   ?occupation wdt:P106 wd:Q33999.
      ?entitiy wdt:P569 ?birthDate. # save birth date
      OPTIONAL {?entitiy wdt:P570 ?deathDate.} # save death date
      ?entitiy wdt:P21 ?gender.
      ?entitiy wdt:P19 ?cityOfBirth.
      ?cityOfBirth wdt:P17 ?contry.
      ?contry wdt:P37 ?officialLang.
      ?officialLang wdt:P424 ?langcode.
      FILTER(?langcode = "he")
      BIND(YEAR(?birthDate) AS ?birthYear)  # save birth year
      BIND(YEAR(?deathDate) AS ?deathYear)  # save birth year
      FILTER(?birthYear > 1500)
      ?entitiy rdfs:label ?o_en.
      FILTER(LANG(?o_en) = "en").
      ?entitiy rdfs:label ?o_he.
      FILTER(LANG(?o_he) = "he").
      ?entitiy rdfs:label ?o_fr.
      FILTER(LANG(?o_fr) = "fr").
    #   ?entitiy rdfs:label ?o_de.
    #   FILTER(LANG(?o_de) = "de").
      ?entitiy rdfs:label ?o_es.
      FILTER(LANG(?o_es) = "es").
      ?entitiy rdfs:label ?o_ar.
      FILTER(LANG(?o_ar) = "ar").
      ?entitiy rdfs:label ?o_ru.
      FILTER(LANG(?o_ru) = "ru").
      ?entitiy rdfs:label ?o_ja.
      FILTER(LANG(?o_ja) = "ja").
    } LIMIT 1000
    """
    sparql = SPARQL("https://query.wikidata.org/sparql")
    qlod = sparql.queryAsListOfDicts(sparqlQuery)
    csv = CSV.toCSV(qlod)
    df = pd.read_csv(io.StringIO(csv))
    print(df)