import pandas as pd

import EntityLinking.FinetuningDatasets.EntityStats as EntityStats


class TransferStats:

    def __init__(self):
        pass


def main():
    count = 0
    df = pd.read_csv("LingualTransfer/Data/query_hebrew_poets_birth_year.csv")
    df["en_views"] = -1
    df["he_views"] = -1
    df["en_q"] = ""
    df["he_q"] = ""
    for index, row in df.iterrows():
        # if count >= 5:
        #     break
        entity_id = row["item"].split("/")[-1]
        df.at[index, "en_views"] = EntityStats.get_daily_average_page_view(entity_id, 'en')
        df.at[index, "he_views"] = EntityStats.get_daily_average_page_view(entity_id, 'he')
        df.at[index, "en_q"] = "What year was " + df.at[index, "itemLabel"] + " born?"
        df.at[index, "he_q"] = "באיזה שנה נולד " + df.at[index, "item_he"] + " ?"
        count += 1
    df.to_csv("LingualTransfer/Data/query_hebrew_poets_birth_year_details.csv")
