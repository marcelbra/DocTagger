import json
import nltk
import spacy
from pred_ner import Predictor
from typing import Dict, Any, List, Optional

class DocBuilder:

    def __init__(self,
                 tokenizer,
                 trainer,
                 _id: int,
                 spacy_dir: str,
                 save_path_dir: str,
                 mode: str="tokenized"
                 ):
        """
        Formats a given document (passed by path to get_doc).
        Extracts the abstract and text passages of CORD-19 dataset.
        Retrieves the BioNER tags of a document in tag_document.

        :param tokenizer: The BERT word tokenizer to be passed to the predictor
        :param trainer: The huggingface trainer used for inference to be passed to the predictor
        :param _id: The ID of this process to save progress accordingly
        :param mode: The mode we expect our DocBuilder to format the data
        """
        self.sent_tokenizer = nltk.sent_tokenize
        self.word_tokenizer = spacy.load(spacy_dir)
        self.pred = Predictor(tokenizer, trainer)
        self._id = _id
        self.mode = mode
        self.save_path = save_path_dir + str(self._id)

    def get_saved_point(self):
        """Loads progress of current DocBuilder."""
        try:
            with open(self.save_path, 'r', encoding="utf-8") as file:
                return int(file.readline())
        except FileNotFoundError:
            self.save(0)
            return 0

    def save(self, i: int):
        """Saves progress of current DocBuilder progress."""
        with open(self.save_path, 'w', encoding="utf-8") as file:
            file.write(str(i))
        return i


    def get_doc(self, path: str) -> (Dict[str: Any],
                                     Dict[str: Any]
                                     ):
        """
        Annotates a given document (path).
        :param path: The path to file
        :return: The tuple of old data and new data
        """
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
    def count_tags(tgd_doc: List[List[str]]) -> int:
        """Counts how many gene tags are in a tagged list."""
        return sum([x.count("B") for x in tgd_doc])

    @staticmethod
    def count_words(tok_abs: List[List[str]],
                    tok_txt: List[List[str]]
                    ) -> int:
        """Counts how many gene tags are in a tagged list."""
        return sum(map(len, tok_abs + tok_txt))

    @staticmethod
    def calc_ratio(amt_tags: int, amt_words: int) -> float:
        """Calculates the percentage of genetags in a document."""
        return float(amt_tags/amt_words)

    @staticmethod
    def write_doc(path: str,
                  data: Dict[str: Any],
                  doc: Dict[str: Any]
                  ):
        """Concatenates original data and new data and writes to file."""
        with open(path, 'w', encoding="utf-8") as f:
            json.dump({**data, **doc}, f)

    def tag_document(self,
                     tok_abs: List[List[str]],
                     tok_txt: List[List[str]]
                     ) -> List[List[str]]:
        """Sets the data, predicts and returns list of tags.
        The structure of the tag list corresponds 1-to-1 to the structure
        of the concatenated tokenized abstract and text.
        """
        self.pred.set_data(tok_abs + tok_txt)
        tags = self.pred.predict()
        return tags

    def get_text_of(self, data: Dict[str: Any]
                    ) -> Optional[str, List[str], List[List[str]]]:
        """Formats the text of a CORD-19 JSON"""

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

    def get_abstract_of(self, data: Dict[str: Any]
                        ) -> Optional[str, List[str], List[List[str]]]:
        """Formats the abstract of a CORD-19 JSON"""

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