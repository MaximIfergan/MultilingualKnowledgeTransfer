from Model import MLCBQA_Model
# import pandas as pd
import EntityLinking.FinetuningDatasets.QA_Linking as QA_Linking
import EntityLinking.FinetuningDatasets.EntityStats as EntityStats
import LingualTransfer.TransferStats as TransferStats
from huggingface_hub import hf_hub_download


def main():
    # QA_Linking.main()
    # EntityStats.main()
    TransferStats.main()
    # MLCBQA_Model.main()

if __name__ == "__main__":
    main()