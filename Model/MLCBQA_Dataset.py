import torch
from torch.utils.data import Dataset


class MLCBQA_Dataset(Dataset):
    """
    Multilingual close book question answering dataset
    """

    def __init__(self, dataframe, tokenizer, source_len, target_len, source_col, target_col,
                 pad_to_max_length=True):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = self.data[target_col]
        self.source_text = self.data[source_col]
        self.pad_to_max_length = pad_to_max_length

    def __len__(self):
        """returns the length of dataset"""
        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        source_text = str(self.source_text.iloc[index])
        target_text = str(self.target_text.iloc[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=self.pad_to_max_length,
            truncation=True,
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=self.pad_to_max_length,
            truncation=True,
            return_tensors="pt",
        )

        source_ids = source["input_ids"].squeeze().to(dtype=torch.long)
        source_mask = source["attention_mask"].squeeze().to(dtype=torch.long)
        target_ids = target["input_ids"].squeeze().to(dtype=torch.long)
        # target_mask = target["attention_mask"].squeeze().to(dtype=torch.long)

        return {
            "source_ids": source_ids, "source_mask": source_mask,
            "target_ids": target_ids, "target_ids_y": target_ids,
            "id": str(self.data.iloc[index]["Id"]), "lang": str(self.data.iloc[index]["Language"])
        }
