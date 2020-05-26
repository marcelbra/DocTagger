import torch
import numpy as np
import spacy
import os
from torch import nn
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from utils_ner import NerDataset, Split, InputExample

class Predictor:

    def __init__(self, tokenizer, splitter, trainer):

        cd_dir = "/home/marcel/Desktop/transformers-master/examples/token-classification"
        # self.pretrained_dir = cd_dir + "/biobert_v1.1_pubmed"
        # self.cache_dir = cd_dir + "/BioBERT_ner/data"
        # self.config_dir = cd_dir + "/biobert_v1.1_pubmed/config.json"
        # self.model_dir = cd_dir + "/BioBERT_ner/output_3/pytorch_model.bin"
        # self.data_dir = cd_dir + "/BioBERT_ner/pred_data"
        # self.labels_dir = cd_dir + "/BioBERT_ner/data/labels.txt"
        # self.output_dir = cd_dir + "/BioBERT_ner"
        self.label_map = {0: "O", 1: "B"}
        self.cache = [cd_dir + "/BioBERT_ner/pred_data/cached_pred_BertTokenizer_256.lock",
                      cd_dir + "/BioBERT_ner/pred_data/cached_pred_BertTokenizer_256"]

        self.tokenizer = tokenizer #AutoTokenizer.from_pretrained(self.pretrained_dir) #, cache_dir=cache_dir)
        self.splitter = splitter #spacy.load(cd_dir + "/en_core_sci_md-0.2.4/en_core_sci_md/en_core_sci_md-0.2.4")
        # self.model = AutoModelForTokenClassification.from_pretrained(self.pretrained_dir, config=self.config_dir) #,cache_dir=cache_dir)
        # self.model.load_state_dict(torch.load(self.model_dir))
        self.trainer = trainer #Trainer(model=self.model, args=TrainingArguments(output_dir=self.output_dir))
        self.data = None

    def align_predictions(self, predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(self.label_map[label_ids[i][j]])
                    preds_list[i].append(self.label_map[preds[i][j]])
        return preds_list, out_label_list

    def set_data(self, tok_sents):
        examples = []
        for guid, sent in enumerate(tok_sents):
            words = [x + "\n" for x in sent]
            labels = ["O" for x in range(len(sent))]
            examples.append(InputExample(guid=f"pred-{guid}", words=words, labels=labels))

        data = NerDataset(
            tokenizer=self.tokenizer,
            examples=examples,
            labels=["B", "O"],
            model_type="BertForTokenClassification",
            max_seq_length=256,
            mode=Split.pred
        )

        self.data = data

    def predict(self):
        predictions, label_ids, metrics = self.trainer.predict(self.data)
        preds_list, _ = self.align_predictions(predictions, label_ids)
        # remove file cache
        for file in self.cache:
            try:
                os.remove(file)
            except OSError:
                pass
        return preds_list