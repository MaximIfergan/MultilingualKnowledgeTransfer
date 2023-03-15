# from Model import MLCBQA_Model
# import pandas as pd
import EntityLinking.FinetuningDatasets.QA_Linking as QA_Linking
import EntityLinking.FinetuningDatasets.EntityStats as EntityStats
import LingualTransfer.TransferStats as Find_a_proper_name
from huggingface_hub import hf_hub_download


def main():
    # QA_Linking.main()
    EntityStats.main()

if __name__ == "__main__":
    main()
    # hf_hub_download(repo_id="nkandpa2/pretraining_entities", filename="wikipedia_entity_map.npz", repo_type="dataset", cache_dir="EntityLinking/PretrainingDatasets")
