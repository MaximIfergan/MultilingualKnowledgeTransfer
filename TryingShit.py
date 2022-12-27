import json
import pandas as pd

# from tqdm.auto import tqdm
#
# # for i in tqdm(range(100000000)):
# #     pass
#
# a = [json.loads(line) for line in open("Datasets/Mintaka/mintaka_dev.json", 'r', encoding='utf-8')]
# print()


def print_title(title):
    res = "      " + title + "      "
    while (len(res) < 90):
        res = "=" + res + "="
    print("# " + res)

print_title("Load models:")

# print_title("Preprocess dataset details:")

# df = pd.read_csv('Datasets/PreprocessDataset.csv')
# print(df[df["Type"] == "train"].head().reset_index(drop=True)[["Question", "Answer"]])
