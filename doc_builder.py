import json
import nltk
import spacy
from pred_ner import Predictor


class DocBuilder:

    def __init__(self, tokenizer, splitter, trainer, mode="tokenized"):
        path = "/home/marcel/Desktop/transformers-master/examples/token-classification/en_core_sci_md-0.2.4/en_core_sci_md/en_core_sci_md-0.2.4"
        self.word_tokenizer = spacy.load(path)
        self.sent_tokenizer = nltk.sent_tokenize
        self.pred = Predictor(tokenizer, splitter, trainer)
        self.mode = mode

    def get_doc(self, path: str):
        with open(path) as file:
            data = json.load(file)
            tok_abs = self.get_abstract_of(data)
            tok_txt = self.get_text_of(data)
            tgd_doc = self.tag_document(tok_abs, tok_txt)
            amt_tags = self.count_tags(tgd_doc)
            amt_words = self.count_words(tok_abs, tok_txt)
            ratio = self.calc_ratio(amt_tags, amt_words)
            doc = {
                "tokenized_abstract": tok_abs,
                "tokenized_text": tok_txt,
                "tagged_document": tgd_doc,
                "amount_of_tags": amt_tags,
                "amount_of_words": amt_words,
                "gene_to_word_ratio": ratio
            }
            return data, doc

    @staticmethod
    def count_tags(tgd_doc):
        return sum([x.count("B") for x in tgd_doc])

    @staticmethod
    def count_words(tok_abs, tok_txt):
        return sum(map(len, tok_abs + tok_txt))

    @staticmethod
    def calc_ratio(amt_tags: int, amt_words: int):
        return float(amt_tags/amt_words)        

    def tag_document(self, tok_abs, tok_txt):
        self.pred.set_data(tok_abs + tok_txt)
        tags = self.pred.predict()
        return tags

    def write_doc(self, path: str, data, doc):
        with open(path, 'w', encoding="utf-8") as f:
            json.dump({**data, **doc}, f)
            
    def get_text_of(self, data):

        # raw text string
        ps = [p["text"] for p in data["body_text"]]
        raw_text = " ".join(ps)
        if self.mode == "raw_string":
            return raw_text

        # list of sentences
        sents = self.sent_tokenizer(raw_text)
        if self.mode == "sentence_list":
            return sents

        # list of tokenized sentences
        sents = [self.word_tokenizer(sent) for sent in sents]
        tokenized = [[str(word) for word in sent] for sent in sents]
        if self.mode == "tokenized":
            return tokenized

    def get_abstract_of(self, data):
        """Expects a document path and returns a list of list of words."""
        raw_abst = data["abstract"]
        abstract = [raw_abst[i]["text"] for i, _ in enumerate(raw_abst)]
        if len(abstract) == 0: return []  # some have no abstract

        # raw abstract string
        abstract = abstract[0]
        if self.mode == "raw_string":
            return abstract

        # list of sentences
        sents = self.sent_tokenizer(abstract)
        if self.mode == "sentence_list":
            return sents

        # list of tokenized sentences
        sents = [self.word_tokenizer(sent) for sent in sents]
        tokenized = [[str(word) for word in sent] for sent in sents]
        if self.mode == "tokenized":
            return tokenized