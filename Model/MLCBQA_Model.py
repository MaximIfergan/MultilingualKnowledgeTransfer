from __future__ import print_function
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os
from rich.table import Column, Table
from rich import box
from rich.console import Console
from Model.MLCBQA_Dataset import MLCBQA_Dataset
import sentencepiece
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
import wandb
from collections import Counter
import string
import re
import pickle

# ===============================      Global Variables:      ===============================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CACHE_DIR = "/cs/labs/oabend/maximifergan/MultilingualKnowledgeTransfer/downloaded_models"

# STDOUT Logger init:
CONSOLE = Console(record=True)
LOGGER = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

# ===============================      Global Functions:      ===============================


def mean_decoder_embedding(embedding):
    result = []
    for layer_index in range(len(embedding[0])):
        layer_mean = torch.zeros_like(embedding[0][layer_index])
        for token_index in range(len(embedding)):
            layer_mean += embedding[token_index][layer_index]
        layer_mean /= len(embedding)
        result.append(layer_mean)
    return result


def mean_encoder_embedding(embedding):
    result = [torch.mean(layer, dim=1) for layer in embedding]
    return result


def train(epoch, tokenizer, model, loader, optimizer):
    """
    Function trains the model with the parameters passed from main function
    """

    model.train()

    # Batch training loop:
    for _, data in enumerate(loader, 0):

        # Batch inputs and labels:
        labels_ids = data["target_ids"].to(DEVICE, dtype=torch.long)
        labels_ids[labels_ids == tokenizer.pad_token_id] = -100  # Change pad token with -100 (HG T5 documentation)
        inputs_ids = data["source_ids"].to(DEVICE, dtype=torch.long)
        mask = data["source_mask"].to(DEVICE, dtype=torch.long)

        # Loss on batch:
        outputs = model(input_ids=inputs_ids, attention_mask=mask, labels=labels_ids)
        loss = outputs.loss

        # Update Loggers:
        if _ % 1000 == 0:
            CONSOLE.print(f"= Epoch: {epoch:2d}  = Step: {_:5d}  = Loss: {loss.item():.4f}")
        if _ % 500 == 0:
            wandb.log({"Training Loss": loss.item()})

        # Update Model weighs:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(epoch, tokenizer, model, loader):
    """
    Validate the models predictions on dev set
    """
    model.eval()
    loss = 0
    step = 0
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            # Batch inputs and labels:
            labels_ids = data["target_ids"].to(DEVICE, dtype=torch.long)
            labels_ids[labels_ids == tokenizer.pad_token_id] = -100  # Change pad token with -100 (HG T5 documentation)
            inputs_ids = data["source_ids"].to(DEVICE, dtype=torch.long)
            mask = data["source_mask"].to(DEVICE, dtype=torch.long)

            # Loss on batch:
            outputs = model(input_ids=inputs_ids, attention_mask=mask, labels=labels_ids)
            loss += outputs.loss.item()
            step += 1
    loss = loss / step
    CONSOLE.print(f"= Epoch: {epoch:2d} Validation Loss: {loss:.4f}\n")
    wandb.log({"Validation Loss": loss})
    return loss


def evaluate(tokenizer, model, loader, save_embedding=False):
    """
    Evaluate the models predictions on dev set
    """
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            # if _ % 1000 == 0:
            #     print(_)
            y = data['target_ids'].to(DEVICE, dtype=torch.long)
            ids = data['source_ids'].to(DEVICE, dtype=torch.long)
            mask = data['source_mask'].to(DEVICE, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
                     generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in y]
            predictions.extend(preds)
            actuals.extend(target)

    result = evaluate_metrics(actuals, predictions)
    CONSOLE.print(f"Evaluation results: Exact_match {result['exact_match']}, F1: {result['f1']}")
    wandb.log({"Exact Match": result['exact_match'], "F1": result['f1']})
    return predictions, actuals, result['f1_scores'], result['exact_match_scores']


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    if (prediction == ""):
        return False
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_metrics(gold_answers, predictions):
    f1 = exact_match = total = 0
    f1_scores = []
    exact_match_scores = []
    for ground_truth, prediction in zip(gold_answers, predictions):
        total += 1
        example_em = exact_match_score(prediction, ground_truth)
        exact_match += example_em
        exact_match_scores.append(example_em)
        # exact_match += metric_max_over_ground_truths(
        #     exact_match_score, prediction, ground_truths)
        example_f1 = f1_score(prediction, ground_truth)
        f1 += example_f1
        f1_scores.append(example_f1)
        # f1 += metric_max_over_ground_truths(
        #     f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1, 'f1_scores': f1_scores, 'exact_match_scores': exact_match_scores}


def save_embedding_layers(tokenizer, model, dataset, source_col, target_col, output_path):

    dataset_size = dataset.shape[0]

    embedding_layers = dict()
    val_set = MLCBQA_Dataset(dataset, tokenizer, None, None, source_col, target_col, pad_to_max_length=False)
    val_params = {"batch_size": 1, "shuffle": False, "num_workers": 0}
    val_loader = DataLoader(val_set, **val_params)

    count = 0
    with torch.no_grad():
        for _, data in enumerate(val_loader, 0):

            if count % 1000 == 0:
                print(f"evaluation at: {round(100 * count / dataset_size)}%")

            ids = data['source_ids'].to(DEVICE, dtype=torch.long).to(DEVICE)
            mask = data['source_mask'].to(DEVICE, dtype=torch.long).to(DEVICE)

            out = model.generate(
                input_ids=ids,
                attention_mask=mask,
                repetition_penalty=2.5,
                num_beams=2,
                length_penalty=1.0,
                early_stopping=True,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            if data["id"][0] not in embedding_layers:
                embedding_layers[data["id"][0]] = dict()


            saved_decoder_hidden_states = [layer[::2] for layer in out.decoder_hidden_states]
            saved_decoder_hidden_states = mean_decoder_embedding(saved_decoder_hidden_states)
            saved_encoder_hidden_states = mean_encoder_embedding(out.encoder_hidden_states[::2])
            embedding_layers[data["id"][0]][data["lang"][0]] = {"encoder_hidden_states": saved_encoder_hidden_states,
                                                                "decoder_hidden_states": saved_decoder_hidden_states}

            count += 1

    with open(output_path, 'wb') as fp:
        pickle.dump(embedding_layers, fp)


def MT5Trainer(dataframe, source_text, target_text, model_params, output_dir="./SavedModels/"):
    """
    MT5 trainer
    """

    # Init wandb logger:
    wandb.login(key="5028a3fdc48caac16f85893a6e275eb36bb8eba5")
    wandb.init(project="CloseBookMultilingualQA", entity="huji-google-collaboration")
    wandb.config = {
        "Model": model_params["MODEL"],
        "learning_rate": model_params["LEARNING_RATE"],
        "epochs": model_params["TRAIN_EPOCHS"],
        "batch_size": model_params["VALID_BATCH_SIZE"],
        "SEED": model_params["SEED"]
    }

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(model_params["SEED"])  # pytorch random seed
    np.random.seed(model_params["SEED"])  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # Loading model and tokenizer:
    CONSOLE.log(f"""[Model]: Loading {model_params["MODEL"]}""")  # STDOUT logger
    # tokenizer = MT5Tokenizer.from_pretrained(model_params["MODEL_DIR"], cache_dir=CACHE_DIR)
    # model = MT5ForConditionalGeneration.from_pretrained(model_params["MODEL_DIR"], cache_dir=CACHE_DIR)
    tokenizer = MT5Tokenizer.from_pretrained(model_params["MODEL_DIR"])
    model = MT5ForConditionalGeneration.from_pretrained(model_params["MODEL_DIR"])
    model = model.to(DEVICE)
    CONSOLE.log(f"""[Model]: Model parameters:""")
    CONSOLE.log(model_params)
    CONSOLE.log(f"""[Model]: Loading Completed""")  # STDOUT logger

    # Building dataset:
    CONSOLE.log(f"[Dataset]: Building dataset")
    train_dataset = dataframe[dataframe['DataType'] == "train"].reset_index(drop=True)[[source_text, target_text]]
    val_dataset = dataframe[dataframe['DataType'] == "dev"].reset_index(drop=True)[[source_text, target_text]]

    CONSOLE.print(f"== Datasets Details: ==")
    CONSOLE.print(f"FULL Dataset: {dataframe.shape}")
    CONSOLE.print(f"TRAIN Dataset: {train_dataset.shape}")
    CONSOLE.print(f"TEST Dataset: {val_dataset.shape}\n")

    training_set = MLCBQA_Dataset(train_dataset, tokenizer, model_params['MAX_SOURCE_TEXT_LENGTH'],
                                  model_params['MAX_TARGET_TEXT_LENGTH'], source_text, target_text)
    val_set = MLCBQA_Dataset(val_dataset, tokenizer, model_params['MAX_SOURCE_TEXT_LENGTH'],
                                  model_params['MAX_TARGET_TEXT_LENGTH'], source_text, target_text)

    train_params = {"batch_size": model_params["TRAIN_BATCH_SIZE"], "shuffle": True, "num_workers": 4}
    val_params = {"batch_size": model_params["VALID_BATCH_SIZE"], "shuffle": False, "num_workers": 4}
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    CONSOLE.log(f"[Dataset]: Building Completed.")

    CONSOLE.log(f"[Finetuning]: Finetuning Model")
    # Init optimizer:
    optimizer = torch.optim.Adam(params=model.parameters(), lr=model_params["LEARNING_RATE"])

    # max_loss = float("inf")

    # Training loop:
    for epoch in range(model_params["TRAIN_EPOCHS"]):
        train(epoch, tokenizer, model, training_loader, optimizer)
        loss = validate(epoch, tokenizer, model, val_loader)
        # if loss < max_loss and epoch != (model_params["TRAIN_EPOCHS"] - 1):
        #     max_loss = loss
        path = os.path.join(output_dir, f"model-epoch-{epoch + 1}")
        model.save_pretrained(path)
    CONSOLE.log(f"[Fine-tuning]: Completed.")

    # Save model:
    CONSOLE.log(f"[Model]: Saving Model")
    path = os.path.join(output_dir, "model_files")
    model.save_pretrained(path)

    # Evaluate model:
    CONSOLE.log(f"[Evaluation]: Initiating Evaluation")
    predictions, actuals, f1_scores, em_scores = evaluate(tokenizer, model, val_loader)
    final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals, "F1": f1_scores, "EM": em_scores})
    final_df.to_csv(os.path.join(output_dir, "predictions_extract.csv"))
    CONSOLE.log(f"[Evaluation]: Evaluation Completed")

    CONSOLE.save_text(os.path.join(output_dir, "logs.txt"))


def main():

    # After training path: "/cs/labs/oabend/maximifergan/MKT/SavedModels/mT5-base-2-epochs/model_files/"

    # model_params = {
    #     "MODEL": "mt5-small",
    #     "MODEL_DIR": "google/mt5-small",
    #     "TRAIN_BATCH_SIZE": 8,
    #     "VALID_BATCH_SIZE": 8,
    #     "TRAIN_EPOCHS": 6,
    #     "LEARNING_RATE": 1e-4,
    #     "MAX_SOURCE_TEXT_LENGTH": 90,
    #     "MAX_TARGET_TEXT_LENGTH": 312,
    #     "SEED": 18,
    # }

    # All Answers:
    # MAX_SOURCE_TEXT_LENGTH : 90
    # MAX_TARGET_TEXT_LENGTH : 312

    # Normal Answers:
    # MAX_SOURCE_TEXT_LENGTH : 90
    # MAX_TARGET_TEXT_LENGTH : 312

    # === For checking training pipeline ===
    # df = pd.read_csv("Data/Datasets/PreprocessDatasetAllLangs.csv").sample(frac=1)[:80]

    # df = pd.read_csv("Data/Datasets/PreprocessDatasetAnswerAll.csv")
    # output_dir = "Model/SavedModels/mT5-base-all-answers"
    # os.makedirs(output_dir)
    #
    # MT5Trainer(
    #     dataframe=df,
    #     source_text="Question",
    #     target_text="answer_all",
    #     model_params=model_params,
    #     output_dir=output_dir,
    # )

    # ========================= Check TEXT_LENGTH of dataset: =========================
    # df = pd.read_csv("Data/Datasets/PreprocessDatasetAllLangs.csv")
    # tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    # source_max = 0
    # target_max = 0
    # for index, row in df.iterrows():
    #     s = tokenizer.batch_encode_plus(
    #         [str(row["Question"])],
    #         return_tensors="pt")
    #     t = tokenizer.batch_encode_plus(
    #         [str(row["Answer"])],
    #         return_tensors="pt")
    #     source_max = source_max if s["input_ids"].squeeze().shape[0] <= source_max else s["input_ids"].squeeze().shape[0]
    #     target_max = target_max if t["input_ids"].squeeze().shape[0] <= target_max else t["input_ids"].squeeze().shape[0]
    # print(source_max)
    # print(target_max)

    # ===================================== eval specific epochs =====================================
    # dir = "/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/mT5-base/model-epoch-0"
    # print("load model")
    # model = MT5ForConditionalGeneration.from_pretrained(dir).to(DEVICE)
    # tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base")
    # print("finish load model")
    # print("bulid val set")
    # val_dataset = df[df['DataType'] == "dev"].reset_index(drop=True)[["Question", "Answer"]]
    # val_set = MLCBQA_Dataset(val_dataset, tokenizer, 396, 32)
    # val_params = {"batch_size": 4, "shuffle": False, "num_workers": 0}
    # val_loader = DataLoader(val_set, **val_params)
    # print("finish bulid val set")
    # print("start val")
    # predictions, actuals, f1_scores, em_scores = evaluate(tokenizer, model, val_loader)
    # print("end val")
    # print("save res")
    # final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals, "F1": f1_scores, "EM": em_scores})
    # final_df.to_csv(os.path.join(dir, "predictions_extract-bug.csv"))
    # print("end save res")

    # ===============================      save embeddings:      ===============================
    pred_dir = "/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-large/mT5-large-continue/predictions.csv"
    predictions = pd.read_csv(pred_dir)
    df = pd.read_csv("Data/Datasets/PreprocessDatasetAllLangs.csv")
    df = df.loc[df['DataType'] == "dev"]
    df["Prediction"] = list(predictions["Generated Text"])
    df["F1"] = list(predictions["F1"])
    df["EM"] = list(predictions["EM"])
    df = df.loc[df['Dataset'] != "NQ"]
    df["Know"] = 0
    ids = list(df.loc[df['F1'] > 0.5]["Id"].unique())
    for id in ids:
        df.loc[df['Id'] == id, 'Know'] = 1
    df = df.loc[df['Know'] == 1]

    dir = "/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-large/mT5-large-continue/model_files"
    model_name = "mT5-large"
    model = MT5ForConditionalGeneration.from_pretrained(dir).to(DEVICE)
    tokenizer = MT5Tokenizer.from_pretrained("google/mT5-large")
    save_embedding_layers(tokenizer, model, df, "Question", "Answer", f'/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-large/mT5-large-continue/embedding_layers_{model_name}.pkl')



    # =========================      Debug saving the embeddings:      =========================
    pred_dir = "/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-large/mT5-large-continue/predictions.csv"
    predictions = pd.read_csv(pred_dir)
    df = pd.read_csv("Data/Datasets/PreprocessDatasetAllLangs.csv")
    df = df.loc[df['DataType'] == "dev"]
    df["Prediction"] = list(predictions["Generated Text"])
    df["F1"] = list(predictions["F1"])
    df["EM"] = list(predictions["EM"])
    df = df.loc[df['F1'] > 0.5]
    df = df.loc[df['Dataset'] != "NQ"]

    dir = "/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-large/mT5-large-continue/model_files"
    model_name = "mT5-large"
    model = MT5ForConditionalGeneration.from_pretrained(dir).to(DEVICE)
    tokenizer = MT5Tokenizer.from_pretrained("google/mT5-large")
    save_embedding_layers(tokenizer, model, df, "Question", "Answer", f'/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-large/mT5-large-continue/embedding_layers_{model_name}.pkl')

















    # sent = "today everything in fine"
    # tokenizer = MT5Tokenizer.from_pretrained("google/mT5-small")
    # model = MT5ForConditionalGeneration.from_pretrained("google/mT5-small")
    #
    # source = tokenizer.batch_encode_plus(
    #     [sent],
    #     max_length=None,
    #     pad_to_max_length=False,
    #     truncation=True,
    #     return_tensors="pt",
    # )
    #
    # for g in list(source["input_ids"][0]):
    #     print("============")
    #     print(g)
    #     print(tokenizer.decode(g))
    #     print("============")
    #
    # out = model.generate(
    #     input_ids=source["input_ids"],
    #     attention_mask=source['attention_mask'],
    #     repetition_penalty=2.5,
    #     num_beams=2,
    #     length_penalty=1.0,
    #     early_stopping=True,
    #     output_hidden_states=True,
    #     return_dict_in_generate=True
    # )
    #
    # pred_dir = "/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-base/predictions.csv"
    # predictions = pd.read_csv(pred_dir)
    # df = pd.read_csv("Data/Datasets/PreprocessDatasetAllLangs.csv")
    # df = df.loc[df['DataType'] == "dev"]
    # df["Prediction"] = list(predictions["Generated Text"])
    # df["F1"] = list(predictions["F1"])
    # df["EM"] = list(predictions["EM"])
    # df = df.loc[df['F1'] > 0.5]
    # df = df[df['Dataset'] != "NQ"]
    # correct_ids = set(df["Id"].unique())
    #
    # with open('/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-base/embedding_layers_mT5-base.pkl', 'rb') as fp:
    #     big_dict = pickle.load(fp)
    #
    # smaller_dict = dict()
    # for id in correct_ids:
    #     smaller_dict[id] = dict()
    #     for lang in big_dict[id]:
    #         new_decoder_hidden_states = []
    #         for layer in big_dict[id][lang].decoder_hidden_states:
    #             new_decoder_hidden_states.append(layer[1, 5, 8])
    #         smaller_dict[id][lang] = {"encoder_hidden_states": big_dict[id][lang].encoder_hidden_states[1, 5, 8],
    #                                         "decoder_hidden_states": new_decoder_hidden_states}
    #
    # with open('/home/maxim758/MultilingualKnowledgeTransfer/Model/SavedModels/FinalModels/mT5-base/small_embedding_layers_mT5-base.pkl', 'wb') as fp:
    #     pickle.dump(smaller_dict, fp)
