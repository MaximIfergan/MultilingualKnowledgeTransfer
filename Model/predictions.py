import pandas as pd
import torch

# ===============================      Global Variables:      ===============================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CACHE_DIR = "/cs/labs/oabend/maximifergan/MKT/downloaded_models"

# ===============================      Global Functions:      ===============================


def get_answer(model, tokenizer, question):
    source_encoding = tokenizer(
        question,
        max_length=396,
        padding="max_length",
        truncation="only_second",
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    ids = model.generate(
        input_ids=source_encoding['input_ids'],
        num_beams=1,
        max_length=50,
        use_cache=True)

    preds = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for id in ids]

    return "".join(preds)


def simple_questions(model, tokenizer):
    simple_questions = [("what is the capital of USA?", "Washington, D.C."),
                        ("what is the capital of Israel?", "Jerusalem"),
                        ("what is the capital of Italy?", "Rome"),
                        ("what is the capital of France?", "Paris"),
                        ("what is the capital of Germany?", "Berlin"),
                        ("In which city is the Eiffel Tower located?", "Paris"),
                        ("In which city is the Tower of Pisa located?", "Rome"),
                        ("In what year was the state of Israel established?", "1948"),
                        ("What is the height of the Eiffel Tower in meters?", "300 m, 330 m to tip"),
                        ("How many meters the Eiffel Tower?", "300 m, 330 m to tip"),
                        ("Quelle est la hauteur de la Tour Eiffel?", "300 m, 330 m to tip")]
    for question in simple_questions:
        print(f"Question: {question[0]}")
        print(f"Answer: {question[1]}")
        print(f"Model answer: {get_answer(model, tokenizer, question[0])}")


if __name__ == "__main__":
    df = pd.read_csv('SavedModels/OldModels/mT5-base-6-epochs-lang-intersc/validation_set_with_results_old.csv')
    # print(df.groupby(["Dataset"])["F1", "EM"].mean() * 100)
    # print(df.groupby(["Language"])["F1", "EM"].mean() * 100)
    (df.groupby(["Type"])["F1", "EM"].mean() * 100).round(2).to_csv("Types.csv")
    # df = df.loc[df['Dataset'] != "NQ"]  # only parallel datasets
    # df = df.loc[df['F1'] > 0.5]  # only success answers
    # df["count"] = 1
    # df1 = df.groupby(["Id"])["count"].sum()
    # df1.plot.hist(title='Number of correct languages answers', bins=8)
    # plt.xlabel('Number of languages')
    # plt.ylabel('Count')
    # plt.show()

    # # =================================      Load models:      =================================
    # print("[Loading Tokenizer]:")
    # tokenizer2 = MT5Tokenizer.from_pretrained("SavedModels/mT5-base-2-epochs/model_files", cache_dir=CACHE_DIR)
    # tokenizer6 = MT5Tokenizer.from_pretrained("SavedModels/mT5-base-6-epochs-lang-intersc/model_files", cache_dir=CACHE_DIR)
    #
    # print("[Loading Model]:")
    # model2 = MT5ForConditionalGeneration.from_pretrained("SavedModels/mT5-base-2-epochs/model_files", cache_dir=CACHE_DIR)
    # model6 = MT5ForConditionalGeneration.from_pretrained("SavedModels/mT5-base-6-epochs-lang-intersc/model_files", cache_dir=CACHE_DIR)

    # # # =====================      Generate the validation with results:      =====================
    # #
    # df = pd.read_csv('Datasets/PreprocessDatasetIntersecet.csv')
    # val_dataset = df[df['DataType'] == "dev"]
    # df = pd.read_csv('SavedModels/mT5-base-6-epochs-lang-intersc/predictions.csv')
    # predictions = df['Generated Text'].tolist()
    # actual = df['Actual Text'].tolist()
    # result = MLCBQA_Model.evaluate_metrics(actual, predictions)
    # print(f"total_f1: {result['f1']}, total_em: {result['exact_match']}")
    # val_dataset['Prediction'] = predictions
    # val_dataset['F1'] = result['f1_scores']
    # val_dataset['EM'] = result['exact_match_scores']
    # val_dataset.to_csv(os.path.join("SavedModels/mT5-base-6-epochs-lang-intersc", "validation_set_with_results_old.csv"))