from Model import MLCBQA_Model
import pandas as pd
import EntityLinking.FinetuningDatasets.QA_Linking as QA_Linking


def main():
    QA_Linking.link_finetuning_dataset()


if __name__ == "__main__":
    main()