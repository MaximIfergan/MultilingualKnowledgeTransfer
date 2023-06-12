import pandas as pd
import random
import pandas as pd
# from mwviews.api import PageviewsClient
from jsonlines import jsonlines
from wikidata.client import Client
import pywikibot.data.api as api
import pywikibot
from tqdm.auto import tqdm
import numpy as np
import json
import matplotlib.pyplot as plt
import Data.DataPreprocessing as DataPreprocessing
import sys
import pickle
import re
import os
import EntityLinking.FinetuningDatasets.EntityStats as EntityStats
import Model.MLCBQA_Model as MLCBQA_Model


# ===============================      Global Variables:      ===============================

F1_SUCCESS = 0.5
MKQA_ENTITIES_TO_PV = "EntityLinking/FinetuningDatasets/Results/MKQA_entities_to_pv.pkl"
MINTAKA_ENTITIES_TO_PV = "EntityLinking/FinetuningDatasets/Results/Mintaka_entities_to_pv.pkl"
ID2ENTITIES_PATH = "EntityLinking/FinetuningDatasets/Results/id2entities.pkl"
ENTITY2PV_PATH = "EntityLinking/FinetuningDatasets/Results/entity2pv1.pkl"
DATASET_PATH = "Data/Datasets/PreprocessDatasetAllLangs.csv"
CLIENT = Client()
LANGS_ALL_ANSWERS = ["English:", "Arabic:", "German:", "Japanese:", "Portuguese:", "Spanish:", "Italian:", "French:"]

# ===============================      Global functions:      ===============================


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """ off the shelf from matplotlib examples """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=8)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=8)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="right")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, textcolors=("black", "white"), threshold=None, **textkw):
    """ off the shelf from matplotlib examples """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    if data.shape[0] > 15:
        return []
    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, data[i, j], **kw)
            texts.append(text)
    return texts


def PMF(f_res, s_res):
    im = [0, 1]
    total = f_res.shape[0]
    final_res = 0
    for i in im:
        for j in im:
            px = np.sum(f_res == i) / total
            py = np.sum(s_res == j) / total
            p_x_y = np.sum(np.logical_and(f_res == i, s_res == j)) / (total * total)
            if px == 0 or py == 0 or p_x_y == 0:
                continue
            final_res += (p_x_y * np.log(p_x_y / (py * px)))
    return final_res


def extract_answer_in_lang(all_answers, lang):
    l_all_answers = all_answers.split()
    final_answer = ""
    for i in range(len(l_all_answers)):
        if l_all_answers[i] == (DataPreprocessing.CODE2LANG[lang] + ":"):
            i += 1
            while i < len(l_all_answers) and l_all_answers[i] not in LANGS_ALL_ANSWERS:
                final_answer += (" " + l_all_answers[i])
                i += 1
            break
    return final_answer


def all_answer_to_lang_answer(predictions_path, output_dir="", data_path="Data/Datasets/PreprocessDatasetAnswerAll.csv"):
    data = pd.read_csv(data_path)
    data = data.loc[data['DataType'] == "dev"]
    model_predictions = pd.read_csv(predictions_path)
    model_predictions = list(model_predictions["Generated Text"])
    predictions = []
    actuals = []
    i = 0
    for row_index, row in data.iterrows():
        lang = row["Language"]

        if row["Dataset"] == "NQ":
            actuals.append(row["Answer"])
            predictions.append(model_predictions[i])
            i += 1
            continue
        predictions.append(extract_answer_in_lang(model_predictions[i], lang))
        actuals.append(row["Answer"])
        i += 1
    result = MLCBQA_Model.evaluate_metrics(actuals, predictions)
    f1_scores, em_scores = result['f1_scores'], result['exact_match_scores']
    em_scores = [int(element) for element in em_scores]
    final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals, "F1": f1_scores, "EM": em_scores})
    final_df.to_csv(os.path.join(output_dir, "predictions_extract.csv"))


# ====================================      Class:      ====================================


class TransferStats:
    """ this class produce statistics analysis on the success of the model on the different languages """

    def __init__(self, model_predictions, exp_name, id2entities_path=ID2ENTITIES_PATH, entity2pv_path=ENTITY2PV_PATH,
                 data_path=DATASET_PATH):
        predictions = pd.read_csv(model_predictions)
        self.exp_name = exp_name

        self.results = pd.read_csv(data_path)
        self.results = self.results.loc[self.results['DataType'] == "dev"]
        self.results["Prediction"] = list(predictions["Generated Text"])
        self.results["F1"] = list(predictions["F1"])
        self.results["EM"] = list(predictions["EM"])

        # For old dataset (Intersect)
        # self.results = pd.read_csv('Model/SavedModels/mT5-base-6-ep-inter/validation_set_with_results.csv')

        with open(id2entities_path, "rb") as fp:
            self.id2entities = pickle.load(fp)
        with open(entity2pv_path, "rb") as fp:
            self.entity2pv = pickle.load(fp)

    def evaluation_pipeline(self):
        """ runs the hole evaluation on the predictions """
        self.plot_results_by_dataset()
        self.plot_results_by_type()
        self.plot_results_by_language()
        self.plot_LKB_by_dataset(1)
        self.plot_LKB_by_dataset(2)
        self.plot_LKB_by_type(1)
        self.plot_LKB_by_type(2)
        self.plot_number_of_languages_per_question_by_languages("Mintaka")
        self.plot_number_of_languages_per_question_by_languages("MKQA")
        self.plot_number_of_languages_per_question_by_type("Mintaka")
        # self.plot_number_of_languages_per_question_by_type("MKQA")
        # self.plot_languages_relation_performance_mat("Mintaka")
        # self.plot_languages_relation_performance_mat("MKQA")
        # self.plot_pmf_mat("Mintaka")
        # self.plot_pmf_mat("MKQA")

    def LKB1(self, filter_dataset=None):
        """
        LKB1 on the predictions
        :param filter_dataset: a python dic with in shape of {"col": {}, "value": {}} to filter the dataframe before
               calculating.
        :return: LKB1
        """

        df = self.results
        if filter_dataset:
            df = df[df[filter_dataset["col"]] == filter_dataset["value"]]
        only_correct_answers = df[df['F1'] > F1_SUCCESS]
        correct_ids = set(only_correct_answers["Id"])
        return round(only_correct_answers.shape[0] / (len(correct_ids) * len(df["Language"].unique())), 3)

    def LKB2(self, filter_dataset=None):
        """
        LKB2 on the predictions
        :param filter_dataset: a python dic with in shape of {"col": {}, "value": {}} to filter the dataframe before
               calculating.
        :return: LKB2
        """
        df = self.results
        if filter_dataset:
            df = df[df[filter_dataset["col"]] == filter_dataset["value"]]
        only_correct_answers = df[df['F1'] > F1_SUCCESS]
        correct_ids = set(only_correct_answers["Id"])
        only_correct_answers.loc[:, "count"] = 1
        gb_num_correct_lang = only_correct_answers.groupby(["Id"])["count"].sum()
        num_of_question_that_are_correct_in_all_lang = \
            (gb_num_correct_lang[gb_num_correct_lang == len(df["Language"].unique())]).shape[0]
        return round(num_of_question_that_are_correct_in_all_lang / len(correct_ids), 3)

    def get_success_average_pv(self, failure=False):
        # TODO fix this function to be on a specific dataset
        if failure:
            df = self.results[self.results['F1'] < F1_SUCCESS]
        else:
            df = self.results[self.results['F1'] > F1_SUCCESS]
        average_dict = {}
        count_id = 0
        count_pv = 0
        for lang in DataPreprocessing.FINETUNING_LANGS_INTERSEC:
            average_dict[lang] = []
        for index, row in df.iterrows():
            if row["Id"] in self.id2entities and self.id2entities[str(row["Id"])]:
                entities = self.id2entities[str(row["Id"])]
            else:
                count_id += 1
                continue
            average_pv = []
            for entity in entities["q_entities"] + entities["a_entities"]:
                if entity in self.entity2pv[row["Language"]] and self.entity2pv[row["Language"]][entity][1] != -1:
                    average_pv.append(self.entity2pv[row["Language"]][entity][1])
                else:
                    count_pv += 1
            if not average_pv:
                continue
            average_dict[row["Language"]].append(sum(average_pv) / len(average_pv))
        for lang in DataPreprocessing.FINETUNING_LANGS_INTERSEC:
            # if lang != "ar":
            #     continue
            # average_dict[lang] = round(sum(average_dict[lang]) / len(average_dict[lang]), 2)
            # average_dict[lang] = min(average_dict[lang])
            average_dict[lang] = max(average_dict[lang])
            # average_dict[lang].sort()
            # average_dict[lang] = average_dict[lang][:1000]
            # plt.hist(average_dict[lang], 15)
            # plt.title(lang)
            # plt.show()
        # print(count_pv)
        # print(count_id)
        return average_dict

    def plot_LKB_by_dataset(self, lkb_type=1):
        """ plot LKB {1/2} by dataset """
        lkb = self.LKB1 if lkb_type == 1 else self.LKB2
        datasets = list(self.results["Dataset"].unique())
        datasets.remove("NQ")
        values = list()
        for dataset in datasets:
            values.append(lkb({"col": "Dataset", "value": dataset}))
        datasets = [(datasets[i] + " (" + str(values[i]) + ")") for i in range(len(datasets))]
        fig, ax = plt.subplots()
        ax.bar(datasets, values)
        ax.set_ylabel(f"LKB {lkb_type}")
        ax.set_title(f"LKB {lkb_type} Results by Dataset {self.exp_name}")
        plt.ylim(0, 1)
        plt.show()

    def plot_LKB_by_type(self, lkb_type=1):
        """ plot LKB {1/2} by type """
        lkb = self.LKB1 if lkb_type == 1 else self.LKB2
        types = list(self.results["Type"].unique())
        types.remove("nq")
        values = list()
        for i in range(len(types)):
            values.append(lkb({"col": "Type", "value": types[i]}))
            if types[i] == "number_with_unit":
                types[i] = "number + u"
        types = [types[i] for i in range(len(types))]
        fig, ax = plt.subplots()
        ax.bar(types, values)
        ax.set_xticks(types, types, rotation=45, size="small")
        ax.set_ylabel(f"LKB {lkb_type}")
        ax.set_title(f"LKB {lkb_type} Results by Type {self.exp_name}")
        plt.gcf().subplots_adjust(bottom=0.2)
        plt.ylim(0, 1)
        plt.show()

    def plot_results_by_language(self):
        """ plots the accuracy by language """
        df = self.results[self.results["Dataset"] != "NQ"]
        df = df.groupby(["Language"])["F1", "EM"].mean() * 100
        labels = list(df.axes[0])
        f1 = [round(value, 2) for value in df["F1"]]
        em = [round(value, 2) for value in df["EM"]]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, f1, width, label='F1')
        rects2 = ax.bar(x + width / 2, em, width, label='EM')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title(f'Scores By Language {self.exp_name}')
        ax.set_xticks(x, labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()
        plt.ylim(0, 35)
        plt.show()

    def plot_results_by_dataset(self):
        """ plots the accuracy by datasets """
        all = self.results[["F1", "EM"]].mean() * 100
        df = self.results.groupby(["Dataset"])["F1", "EM"].mean() * 100
        labels = ["All"] + list(df.axes[0])
        f1 = [round(all["F1"], 2)] + [round(value, 2) for value in df["F1"]]
        em = [round(all["EM"], 2)] + [round(value, 2) for value in df["EM"]]
        unique = []
        correct_results = self.results.loc[self.results['F1'] > F1_SUCCESS]  # only success answers
        unique.append(len(correct_results['Id'].unique()) / len(self.results['Id'].unique()))
        for value in list(df.axes[0]):
            unique.append(len(correct_results.loc[(correct_results["Dataset"] == value) &
                                                  (correct_results["Type"] != "comparative") &
                                                  (correct_results["Type"] != "binary") &
                                                  (correct_results["Type"] != "yesno")]['Id'].unique()) /
                          len(self.results.loc[(self.results["Dataset"] == value) &
                                               (self.results["Type"] != "comparative") &
                                               (self.results["Type"] != "binary") &
                                               (self.results["Type"] != "yesno")]['Id'].unique()))
        unique = [round(value * 100, 2) for value in unique]

        x = np.arange(len(labels))  # the label locations
        width = 0.25  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, f1, width, label='F1')
        rects2 = ax.bar(x, em, width, label='EM')
        rects3 = ax.bar(x + width, unique, width, label='Unique')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title(f'Scores By Dataset {self.exp_name}')
        ax.set_xticks(x, labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)
        ax.bar_label(rects3, padding=3)

        fig.tight_layout()

        plt.show()

    def plot_results_by_type(self):
        """ plots the accuracy by type """
        df = self.results[self.results["Dataset"] != "NQ"]
        df = df.groupby(["Type"])["F1", "EM"].mean() * 100
        labels = list(df.axes[0])
        f1 = [round(value, 2) for value in df["F1"]]
        em = [round(value, 2) for value in df["EM"]]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, f1, width, label='F1')
        rects2 = ax.bar(x + width / 2, em, width, label='EM')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title(f'Scores By Type {self.exp_name}')
        ax.set_xticks(x, labels, rotation=45, size="small")
        ax.legend()

        ax.bar_label(rects1, padding=5, size="small")
        ax.bar_label(rects2, padding=2, size="small")

        fig.tight_layout()

        plt.show()

    def plot_number_of_languages_per_question_by_languages(self, dataset):
        """ plots a histogram of the correct questions histogram by the number of correct answers in different languages
            showing for each language proportion in the bar"""
        df = self.results.loc[self.results['Dataset'] == dataset]
        langs = list(df["Language"].unique())

        df = df.loc[df['F1'] > F1_SUCCESS]  # only success answers
        df["correct"] = 1
        dfgb = df.groupby(["Id"])["correct"].sum()
        for i in dfgb.axes[0]:
            df.loc[df['Id'] == i, 'correct'] = dfgb[i]
        df['count'] = 1

        fig, ax = plt.subplots()
        width = 0.35  # the width of the bars: can also be len(x) sequence
        labels = list(range(1, len(langs) + 1))
        sum = np.zeros(len(langs))
        for lang in langs:
            num_of_questions = np.zeros(len(langs))
            dfgbt = df[df["Language"] == lang].groupby('correct')['count'].sum()
            for i in dfgbt.axes[0]:
                num_of_questions[i - 1] = dfgbt[i]
            num_of_questions = num_of_questions / np.array(range(1, len(langs) + 1))
            ax.bar(labels, num_of_questions, width, label=lang, bottom=sum)
            sum += num_of_questions

        ax.set_xticks(list(range(1, len(langs) + 1)), list(range(1, len(langs) + 1)), size="small")
        ax.set_ylabel('# questions')
        ax.set_ylabel('# languages received correct answer')
        ax.set_title(f'{dataset} correct questions histogram by language {self.exp_name}', fontsize=11)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
        fig.tight_layout()
        # plt.ylim(0, 450)
        plt.show()

    def plot_number_of_languages_per_question_by_type(self, dataset):
        """ plots a histogram of the correct questions histogram by the number of correct answers in different languages
            showing for each type of QA proportion in the bar"""
        df = self.results.loc[self.results['Dataset'] == dataset]
        df = df.loc[df['F1'] > F1_SUCCESS]  # only success answers
        langs = list(df["Language"].unique())
        types = list(df["Type"].unique())
        df["correct"] = 1
        dfgb = df.groupby(["Id"])["correct"].sum()
        for i in dfgb.axes[0]:
            df.loc[df['Id'] == i, 'correct'] = dfgb[i]
        df['count'] = 1

        fig, ax = plt.subplots()
        width = 0.35  # the width of the bars: can also be len(x) sequence
        labels = list(range(1, len(langs) + 1))
        sum = np.zeros(len(langs))
        for type in types:
            num_of_questions = np.zeros(len(langs))
            dfgbt = df[df["Type"] == type].groupby('correct')['count'].sum()
            for i in dfgbt.axes[0]:
                num_of_questions[i - 1] = dfgbt[i]
            num_of_questions = num_of_questions / np.array(range(1, len(langs) + 1))
            ax.bar(labels, num_of_questions, width, label=type, bottom=sum)
            sum += num_of_questions

        ax.set_xticks(list(range(1, len(langs) + 1)), list(range(1, len(langs) + 1)), size="small")
        ax.set_ylabel('# questions')
        ax.set_ylabel('# languages received correct answer')
        ax.set_title(f'{dataset} correct questions histogram {self.exp_name}', fontsize=9)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 7})
        fig.tight_layout()
        # plt.ylim(0, 300)
        plt.show()

    def plot_types_distribution_for_evaluation_set(self, dataset):
        # TODO test this function!
        df = self.results.loc[self.results['Dataset'] == dataset]
        df["count"] = 1
        df = df.groupby(["Type"])["count"].sum() / len(DataPreprocessing.FINETUNING_LANGS_INTERSEC)
        labels = list(df.axes[0])
        count = [int(t) for t in list(df)]
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots()
        rects = ax.bar(x - width / 2, count, width)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('# questions')
        ax.set_title(f"Types histogram {self.exp_name}")
        ax.set_xticks(x, labels, rotation=45, size="small")
        ax.legend()
        ax.bar_label(rects, padding=3)
        fig.tight_layout()
        plt.show()

    def plot_languages_relation_performance_mat(self, dataset):
        """ plots a heat matrix of the proportion of the success of each language from the QA that was answer correct
            in a different language """
        df = self.results.loc[self.results['Dataset'] == dataset]
        df = df.loc[df['F1'] > F1_SUCCESS]  # only success answers
        ids = list(df["Id"].unique())
        langs = list(df["Language"].unique())
        data = np.zeros((len(ids), len(langs)))
        ref_mat = pd.DataFrame(data, index=ids, columns=langs)
        for lang in langs:
            lang_ids = list(df[df["Language"] == lang]["Id"])
            ref_mat.loc[lang_ids, lang] = 1
        result_mat = pd.DataFrame(columns=langs)
        for lang in langs:
            result_mat.loc[lang] = ref_mat[ref_mat[lang] == 1].sum(axis=0)
            result_mat.loc[lang] = result_mat.loc[lang, :] / result_mat[lang][lang]
        result_mat = result_mat.round(3)

        fig, ax = plt.subplots()
        im, cbar = heatmap(np.array(result_mat), langs, langs, ax=ax,
                           cmap="YlGn", cbarlabel="(% correct answers column from correct row answers")
        texts = annotate_heatmap(im)
        ax.set_title(f"{dataset} languages performance relation on {self.exp_name}")
        fig.tight_layout()
        plt.show()

    def plot_pmf_mat(self, dataset):
        """ plots a heat matrix of the PMF of the sucsess of different langauges """
        df = self.results.loc[self.results['Dataset'] == dataset]
        df = df.loc[df['F1'] > F1_SUCCESS]  # only success answers
        ids = list(df["Id"].unique())
        langs = list(df["Language"].unique())
        data = np.zeros((len(ids), len(langs)))
        ref_mat = pd.DataFrame(data, index=ids, columns=langs)
        for lang in langs:
            lang_ids = list(df[df["Language"] == lang]["Id"])
            ref_mat.loc[lang_ids, lang] = 1
        data = np.zeros((len(langs), len(langs)))
        result_mat = pd.DataFrame(data, index=langs, columns=langs)
        for i in range(len(langs)):
            for j in range(0, len(langs)):
                if j <= i:
                    result_mat.loc[langs[i], langs[j]] = None
                    continue
                result_mat.loc[langs[i], langs[j]] = PMF(np.array(ref_mat[langs[i]]), np.array(ref_mat[langs[j]])) * 1000
        result_mat = result_mat.round(3)

        fig, ax = plt.subplots()
        im, cbar = heatmap(np.array(result_mat), langs, langs, ax=ax,
                           cmap="YlGn", cbarlabel="PMF * 10^3")
        texts = annotate_heatmap(im)
        ax.set_title(f"{dataset} languages success PMF {self.exp_name}")
        fig.tight_layout()
        plt.show()


def main():
    # model_predictions = "Model/SavedModels/mT5-base-6-ep-all-answers/predictions_extract.csv"
    # ts = TransferStats(model_predictions, "mT5-base+", data_path="Data/Datasets/PreprocessDatasetAnswerAll.csv")
    # model_predictions = "Model/SavedModels/mT5-base-6-ep-inter/predictions.csv"
    # ts = TransferStats(model_predictions, "mT5-base", data_path="Model/SavedModels/mT5-base-6-ep-inter/PreprocessDatasetOld.csv")
    model_predictions = "Model/SavedModels/mT5-large-4-ep/predictions.csv"
    ts = TransferStats(model_predictions, "mT5-large")
    ts.evaluation_pipeline()


    # string = "English: John de Mol Arabic: جون دي مول German: John de Mol Japanese: ジョン デ モル Portuguese: John de Mol Spanish: John de Mol Italian: John de Mol French: John de Mol "
    # print(extract_answer_in_lang(string, "fr"))

    # all_answer_to_lang_answer("Model/SavedModels/mT5-base-6-ep-all-answers/predictions.csv")