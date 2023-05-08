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
import EntityLinking.FinetuningDatasets.EntityStats as EntityStats
# import matplotlib as mpl
# mpl.use('TkAgg')  # !IMPORTANT

# ===============================      Global Variables:      ===============================

F1_SUCCESS = 0.5
MKQA_ENTITIES_TO_PV = "EntityLinking/FinetuningDatasets/Results/MKQA_entities_to_pv.pkl"
MINTAKA_ENTITIES_TO_PV = "EntityLinking/FinetuningDatasets/Results/Mintaka_entities_to_pv.pkl"
ID2ENTITIES_PATH = "EntityLinking/FinetuningDatasets/Results/id2entities.pkl"
ENTITY2PV_PATH = "EntityLinking/FinetuningDatasets/Results/entity2pv1.pkl"
DATASET_PATH = "Data/Datasets/PreprocessDatasetAllLangs.csv"
CLIENT = Client()

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
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

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


# ====================================      Class:      ====================================


class TransferStats:

    def __init__(self, model_predictions, exp_name, id2entities_path=ID2ENTITIES_PATH, entity2pv_path=ENTITY2PV_PATH,
                 data_path=DATASET_PATH):
        predictions = pd.read_csv(model_predictions)
        self.exp_name = exp_name
        self.results = pd.read_csv(data_path)
        self.results = self.results.loc[self.results['DataType'] == "dev"]
        self.results["Prediction"] = list(predictions["Generated Text"])
        self.results["F1"] = list(predictions["F1"])
        self.results["EM"] = list(predictions["EM"])
        with open(id2entities_path, "rb") as fp:
            self.id2entities = pickle.load(fp)
        with open(entity2pv_path, "rb") as fp:
            self.entity2pv = pickle.load(fp)

    def evaluation_pipeline(self):
        self.plot_results_by_dataset()
        self.plot_results_by_type()
        self.plot_results_by_language()
        self.plot_number_of_languages_per_question_by_languages("Mintaka")
        self.plot_number_of_languages_per_question_by_languages("MKQA")
        self.plot_number_of_languages_per_question_by_type("Mintaka")
        self.plot_number_of_languages_per_question_by_type("MKQA")
        self.plot_languages_relation_performance_mat("Mintaka")
        self.plot_languages_relation_performance_mat("MKQA")
        self.plot_LKB_by_dataset(1)
        self.plot_LKB_by_dataset(2)
        self.plot_LKB_by_type(1)
        self.plot_LKB_by_type(2)

    def LKB1(self, filter_dataset=None):
        df = self.results
        if filter_dataset:
            df = df[df[filter_dataset["col"]] == filter_dataset["value"]]
        only_correct_answers = df[df['F1'] > F1_SUCCESS]
        correct_ids = set(only_correct_answers["Id"])
        return round(only_correct_answers.shape[0] / (len(correct_ids) * len(df["Language"].unique())), 3)

    def LKB2(self, filter_dataset=None):
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
        plt.show()

    def plot_LKB_by_type(self, lkb_type=1):
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
        plt.show()

    def plot_results_by_language(self):
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

        plt.show()

    def plot_results_by_dataset(self):
        all = self.results[["F1", "EM"]].mean() * 100
        df = self.results.groupby(["Dataset"])["F1", "EM"].mean() * 100
        labels = ["All"] + list(df.axes[0])
        f1 = [round(all["F1"], 2)] + [round(value, 2) for value in df["F1"]]
        em = [round(all["EM"], 2)] + [round(value, 2) for value in df["EM"]]

        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width / 2, f1, width, label='F1')
        rects2 = ax.bar(x + width / 2, em, width, label='EM')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title(f'Scores By Dataset {self.exp_name}')
        ax.set_xticks(x, labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        plt.show()

    def plot_results_by_type(self):
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
        plt.show()

    def plot_number_of_languages_per_question_by_type(self, dataset):
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
        plt.show()

    def plot_types_distribution_for_evaluation_set(self, dataset):
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


def main():
    model_predictions = "Model/SavedModels/mT5-large-4-epochs/predictions.csv"
    ts = TransferStats(model_predictions, "mT5-large")
    ts.evaluation_pipeline()
