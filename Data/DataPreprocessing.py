import json
import pandas as pd
import gzip
# import tensorflow as tf  # when running on CSE -> tensorflow compiler had a lot of errors

# ===============================      Global Variables:      ===============================

FINETUNING_LANGS_INTERSEC = ["en", "ar", "de", "ja", "pt", "es", "it", "fr"]

MINTAKA_LANGS = ["en", "ar", "de", "ja", "pt", "es", "it", "fr", "hi"]

MKQA_LANGS = ["en", "ar", "da", "de", "es", "fi", "fr", "he", "hu", "it", "ja", "km", "ko", "ms", "nl", "no",
                         "pl", "pt", "ru", "sv", "th", "tr", "vi", "zh_cn"]

DATASETS_TYPES = ["binary",
                  "yesno",
                  "comparative",
                  "count",
                  "superlative"
                  "difference",
                  "generic",
                  "ordinal",
                  "intersection",
                  "multihop",
                  "date",
                  "entity",
                  "number",
                  "number_with_unit"]

MAX_WORDS_IN_ANSWER = 10
MKQA_TRAIN_DEV_RATIO = 0.9

# Datasets Paths:
MKQA_PATH = "Datasets/MKQA/mkqa.jsonl"
NQ_TRAIN_PATH_ORG = "Datasets/NQ/v1.0-simplified_simplified-nq-train.jsonl.gz"
NQ_DEV_PATH_ORG = "Datasets/NQ/v1.0-simplified_nq-dev-all.jsonl.gz"
NQ_TRAIN_PATH = "Datasets/NQ/nq_train.txt"
NQ_DEV_PATH = "Datasets/NQ/nq_dev.txt"
MINTAKA_TRAIN_PATH = "Datasets/Mintaka/mintaka_train.json"
MINTAKA_DEV_PATH = "Datasets/Mintaka/mintaka_dev.json"
MINTAKA_TEST_PATH = "Datasets/Mintaka/mintaka_test.json"


# ===============================      Global Functions:      ===============================


# When running on CSE -> tensorflow compiler had a lot of errors (Remove comment when needed)

# def extract_QA_from_NQ(path, out_path):
#     def extract_answer(tokens, span):
#         """Reconstruct answer from token span and remove extra spaces."""
#         start, end = span["start_token"], span["end_token"]
#         ans = " ".join(tokens[start:end])
#
#         # Remove incorrect spacing around punctuation.
#         ans = ans.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
#         ans = ans.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
#         ans = ans.replace("( ", "(").replace(" )", ")")
#         ans = ans.replace("`` ", "\"").replace(" ''", "\"")
#         ans = ans.replace(" 's", "'s").replace("s ' ", "s' ")
#         return ans
#
#     with open(out_path, 'w', encoding="utf-8") as out_file, tf.io.gfile.GFile(path, "rb") as in_file:
#         count = 0
#         for line in gzip.open(in_file):
#             qa_dic = json.loads(line)
#
#             # Remove any examples with more than one answer.
#             if len(qa_dic['annotations'][0]['short_answers']) != 1:
#                 continue
#
#             # Questions in NQ do not include a question mark.
#             question = qa_dic["question_text"] + "?"
#             answer_span = qa_dic['annotations'][0]['short_answers'][0]
#
#             # Handle the two document formats in NQ (tokens or text).
#             if "document_tokens" in qa_dic:
#                 tokens = [t["token"] for t in qa_dic["document_tokens"]]
#             elif "document_text" in qa_dic:
#                 tokens = qa_dic["document_text"].split(" ")
#             answer = extract_answer(tokens, answer_span)
#
#             # Write this line as <example_id>\<question>\t<answer>
#             out_file.write(f"{qa_dic['example_id']}, {question}, {answer}\n")
#
#             count += 1
#             if count % 5000 == 0:
#                 print(count)


# ==================================      Class Code:      ==================================


class DataPreprocessing:

    def __init__(self):
        self.data = None

    def preprocess(self):
        # MKQA:
        mkqa_all = self.preprocess_mkqa(MKQA_PATH, "train")

        # NQ:
        nq_train = self.preprocess_NQ(NQ_TRAIN_PATH, "train")
        nq_dev = self.preprocess_NQ(NQ_DEV_PATH, "dev")

        # Mintaka
        mintaka_train = self.preprocess_Mintaka(MINTAKA_TRAIN_PATH, "train")
        mintaka_dev = self.preprocess_Mintaka(MINTAKA_DEV_PATH, "dev")

        all_rows = mkqa_all + nq_train + mintaka_train + nq_dev + mintaka_dev
        self.data = pd.DataFrame(all_rows)
        self.data = self.data.drop_duplicates(subset=["Id", "Language"])
        # self.data.to_csv('Datasets/PreprocessDataset.csv', encoding='utf-8')

    def preprocess_mkqa(self, path, data_type):
        # Get data from file
        data = [json.loads(line) for line in open(path, 'r', encoding='utf8')]

        # Filter data:
        data = [dic for dic in data if (len(dic['answers']['en']) == 1)]  # remove several answers
        data = [dic for dic in data if
                dic['answers']['en'][0]['type'] not in ['unanswerable', 'long_answer', 'short_phrase']]  # remove unanswerable questions
        data = [dic for dic in data if
                len(dic['answers']['en'][0]['text'].split()) < MAX_WORDS_IN_ANSWER]  # remove long answers
        data_rows = []
        for qa in data:
            for lang in MKQA_LANGS:
                if str(qa['answers'][lang][0]["text"]).replace("\n", "") in ['None', ""]:
                    continue
                question = str(qa['queries'][lang]).replace("\n", "")
                question = question if question[-1] == '?' else question + '?'
                data_rows.append({
                    "Dataset": "MKQA",
                    "DataType": data_type,
                    "Type": qa['answers'][lang][0]["type"],
                    "Id": str(qa["example_id"]),
                    "Language": lang,
                    "Question": str(question),
                    "Answer": str(qa['answers'][lang][0]["text"]).replace("\n", "")
                })

        # Split to dev set
        for i in range(round(len(data_rows) * MKQA_TRAIN_DEV_RATIO), len(data_rows)):
            data_rows[i]["DataType"] = "dev"

        return data_rows

    def preprocess_NQ(self, path, data_type):
        data_rows = []
        with open(path, 'r', encoding='utf8') as fp:
            for line in fp:
                qa = line.split(", ")

                # Filter data:
                if len(qa[2].split()) >= MAX_WORDS_IN_ANSWER:  # remove long answers
                    continue
                question = str(qa[1]).replace("\n", "")
                question = question if question[-1] == '?' else question + '?'
                answer = str(qa[2][:-1]).replace("\n", "")
                if answer in ['None', ""]:
                    continue
                data_rows.append({
                    "Dataset": "NQ",
                    "DataType": data_type,
                    "Type": "nq",
                    "Id": str(qa[0]),
                    "Language": "en",
                    "Question": str(question),
                    "Answer": str(answer)
                })
        return data_rows

    def preprocess_Mintaka(self, path, data_type):
        data_rows = []
        with open(path, 'r', encoding='utf-8') as fp:
            data = json.load(fp)
            for qa in data:

                # Filter:
                if qa['answer']['answer'] is not None and len(qa['answer']['answer']) > 1:
                    continue

                # extract answer
                if qa['answer']['answer'] is None or qa['answer']['answerType'] in ["boolean", "date", "string"]:
                    answer = {lang: qa['answer']['mention'] for lang in FINETUNING_LANGS_INTERSEC}
                elif qa['answer']['answerType'] == "numerical":
                    answer = {lang: qa['answer']['answer'][0] for lang in FINETUNING_LANGS_INTERSEC}
                else:
                    answer = qa['answer']['answer'][0]['label']
                for lang in MINTAKA_LANGS:
                    if lang not in answer:
                        assert lang == "hi"
                        answer[lang] = answer["en"]
                        continue
                    question = str(qa["translations"][lang] if lang != 'en' else qa["question"]).replace("\n", "")
                    question = question if question[-1] == '?' else question + '?'
                    if str(answer[lang]).replace("\n", "") in ['None', ""]:
                        continue
                    data_rows.append({
                        "Dataset": "Mintaka",
                        "DataType": data_type,
                        "Type": qa["complexityType"],
                        "Id": str(qa["id"]),
                        "Language": lang,
                        "Question": str(question),
                        "Answer": str(answer[lang]).replace("\n", "")
                    })
        return data_rows

    def print_dataset_details(self):
        if self.data is None:
            return
        print(f"==========================      Preprocess dataset details:      =========================")
        print(f"------ Number of examples: ------")
        print(f"Total: {self.data.shape[0]}")
        print(f"Training: {self.data.loc[self.data['DataType'] == 'train'].shape[0]}")
        print(f"Training: {self.data.loc[self.data['DataType'] == 'dev'].shape[0]}")
        print(f"NQ: Train: {self.data.loc[(self.data['DataType'] == 'train') & (self.data['Dataset'] == 'NQ')].shape[0]} "
              f"Dev: {self.data.loc[(self.data['DataType'] == 'dev') & (self.data['Dataset'] == 'NQ')].shape[0]}")
        print(f"MKQA: Train: {self.data.loc[(self.data['DataType'] == 'train') & (self.data['Dataset'] == 'MKQA')].shape[0]} "
              f"Dev: {self.data.loc[(self.data['DataType'] == 'dev') & (self.data['Dataset'] == 'MKQA')].shape[0]}")
        print(f"Mintaka: Train: {self.data.loc[(self.data['DataType'] == 'train') & (self.data['Dataset'] == 'Mintaka')].shape[0]} "
              f"Dev: {self.data.loc[(self.data['DataType'] == 'dev') & (self.data['Dataset'] == 'Mintaka')].shape[0]}")


if __name__ == "__main__":
    dp = DataPreprocessing()
    dp.preprocess()
    dp.print_dataset_details()
    dp.data.to_csv("Datasets/PreprocessDatasetAllLangs.csv", encoding='utf-8')  # Save the dataset