import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import DataPreprocessing

class PredictionsStats:

    def __init__(self, data_path="outputs/mT5-base-6-epochs/validation_set_with_results.csv"):
        self.data = pd.read_csv(data_path)

    def plot_results_by_language(self):
        df = self.data[self.data["Dataset"] != "NQ"]
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
        ax.set_title('Scores By Language')
        ax.set_xticks(x, labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        plt.show()

    def plot_results_by_dataset(self):
        all = self.data[["F1", "EM"]].mean() * 100
        df = self.data.groupby(["Dataset"])["F1", "EM"].mean() * 100
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
        ax.set_title('Scores By Dataset')
        ax.set_xticks(x, labels)
        ax.legend()

        ax.bar_label(rects1, padding=3)
        ax.bar_label(rects2, padding=3)

        fig.tight_layout()

        plt.show()

    def plot_results_by_type(self):
        df = self.data[self.data["Dataset"] != "NQ"]
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
        ax.set_title('Scores By Type')
        ax.set_xticks(x, labels, rotation=45, size="small")
        ax.legend()

        ax.bar_label(rects1, padding=5, size="small")
        ax.bar_label(rects2, padding=2, size="small")

        fig.tight_layout()

        plt.show()

    def plot_number_of_languages_per_question_by_languages(self):
        df = self.data.loc[self.data['Dataset'] != "NQ"]  # only parallel datasets
        df = df.loc[df['F1'] > 0.5]  # only success answers
        df["correct"] = 1
        dfgb = df.groupby(["Id"])["correct"].sum()
        for i in dfgb.axes[0]:
            df.loc[df['Id'] == str(i), 'correct'] = dfgb[str(i)]
        df['count'] = 1

        fig, ax = plt.subplots()
        width = 0.35  # the width of the bars: can also be len(x) sequence
        labels = list(range(1, len(DataPreprocessing.FINETUNING_LANGS) + 1))
        sum = np.zeros(len(DataPreprocessing.FINETUNING_LANGS))
        for lang in DataPreprocessing.FINETUNING_LANGS:
            num_of_questions = np.zeros(len(DataPreprocessing.FINETUNING_LANGS))
            dfgbt = df[df["Language"] == lang].groupby('correct')['count'].sum()
            for i in dfgbt.axes[0]:
                num_of_questions[i - 1] = dfgbt[i]
            num_of_questions = num_of_questions / np.array(range(1, len(DataPreprocessing.FINETUNING_LANGS) + 1))
            ax.bar(labels, num_of_questions, width, label=lang, bottom=sum)
            sum += num_of_questions

        ax.set_ylabel('# questions')
        ax.set_ylabel('# languages received correct answer')
        ax.set_title('Correct questions histogram based on the number languages answered correct', fontsize=11)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8})
        fig.tight_layout()
        plt.show()

    def plot_number_of_languages_per_question_by_type(self):
        df = self.data.loc[self.data['Dataset'] != "NQ"]  # only parallel datasets
        df = df.loc[df['F1'] > 0.5]  # only success answers
        df["correct"] = 1
        dfgb = df.groupby(["Id"])["correct"].sum()
        for i in dfgb.axes[0]:
            df.loc[df['Id'] == str(i), 'correct'] = dfgb[str(i)]
        df['count'] = 1

        fig, ax = plt.subplots()
        width = 0.35  # the width of the bars: can also be len(x) sequence
        labels = list(range(1, len(DataPreprocessing.FINETUNING_LANGS) + 1))
        sum = np.zeros(len(DataPreprocessing.FINETUNING_LANGS))
        for type in DataPreprocessing.DATASETS_TYPES:
            num_of_questions = np.zeros(len(DataPreprocessing.FINETUNING_LANGS))
            dfgbt = df[df["Type"] == type].groupby('correct')['count'].sum()
            for i in dfgbt.axes[0]:
                num_of_questions[i - 1] = dfgbt[i]
            num_of_questions = num_of_questions / np.array(range(1,len(DataPreprocessing.FINETUNING_LANGS) + 1))
            ax.bar(labels, num_of_questions, width, label=type, bottom=sum)
            sum += num_of_questions

        ax.set_ylabel('# questions')
        ax.set_ylabel('# languages received correct answer')
        ax.set_title('Correct questions histogram based on the number languages answered correct', fontsize=9)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 7})
        fig.tight_layout()
        plt.show()

    def plot_types_distribution_for_evaluation_set(self):
        df = self.data[self.data["Dataset"] != "NQ"]
        df["count"] = 1
        df = df.groupby(["Type"])["count"].sum() / len(DataPreprocessing.FINETUNING_LANGS)
        labels = list(df.axes[0])
        count = [int(t) for t in list(df)]
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots()
        rects = ax.bar(x - width / 2, count, width)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('# questions')
        ax.set_title('Types histogram')
        ax.set_xticks(x, labels, rotation=45, size="small")
        ax.legend()
        ax.bar_label(rects, padding=3)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    predictionsStats = PredictionsStats()
    predictionsStats.plot_results_by_dataset()
    predictionsStats.plot_results_by_language()
    predictionsStats.plot_results_by_type()
    predictionsStats.plot_number_of_languages_per_question_by_type()
    predictionsStats.plot_number_of_languages_per_question_by_languages()
    predictionsStats.plot_types_distribution_for_evaluation_set()