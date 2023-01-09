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

# df = pd.read_csv('outputs/mT5-base-6-epochs/validation_set_with_results_old.csv')
# df[df["Type"] != "short_phrase"].drop(["Unnamed: 0.1"], axis=1).to_csv("outputs/mT5-base-6-epochs/validation_set_with_results.csv", index=False)
df = pd.DataFrame({'A': [0, 1, 2, 2, 4],
                   'B': [5, 6, 7, 8, 9],
                   'C': ['a', 'b', 'c', 'd', 'e']})
# df[df['C'] == 'd']['A'] = 18
df.loc[df['A'] == 2, 'A'] = 18
print(df)