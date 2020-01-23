import os
import re
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from artm import LDA, BatchVectorizer

from derek.common.io import save_with_pickle, load_with_pickle
from derek.common.vectorizers import AbstractVectorizer
from derek.data.model import Document
from derek.data.processing_helper import StandardTokenProcessor


class SklearnTokenizerAdapter(object):
    def __init__(self, preprocessor: StandardTokenProcessor, token_pattern: str):
        self._processor = preprocessor
        self._token_regex = re.compile(token_pattern)

    def __call__(self, doc: dict) -> List[str]:
        filtered = filter(lambda t: self._token_regex.fullmatch(t) is not None, doc["tokens"])
        return list(map(self._processor, filtered))

    @classmethod
    def from_props(cls, props: dict):
        return cls(StandardTokenProcessor.from_props(props), props.get("token_pattern", r".*"))


class CountVectorizerWrapper(object):
    def __init__(self, config: dict):
        tokenizer = SklearnTokenizerAdapter.from_props(config.get("tokenizer", {}))
        self._vectorizer = CountVectorizer(
            preprocessor=self._identity, tokenizer=tokenizer,
            stop_words='english' if config.get("use_sklearn_stopwords", False) else None,
            ngram_range=tuple(config.get('ngram_range', (1, 1))),
            min_df=config.get('min_df', 1),
            max_df=config.get('max_df', 1.0),
            max_features=config.get('max_features', None)
        )

    @staticmethod
    def _identity(x):
        return x

    def fit_transform(self, docs: List[dict]):
        return self._vectorizer.fit_transform(docs)

    def transform(self, docs: List[dict]):
        return self._vectorizer.transform(docs)

    def get_vocab(self):
        return self._vectorizer.vocabulary_

    def transform_token(self, token: str):
        transformed = self._vectorizer.tokenizer({"tokens": [token]})
        if not transformed or transformed[0] not in self._vectorizer.vocabulary_:
            return None
        return transformed[0]


class LDAModel(object):
    def __init__(self, idx2word, config: dict):
        super().__init__()
        self._idx2word = idx2word
        self._config = config
        self._model: Optional[LDA] = None

    def fit(self, n_dw_matrice):
        bv = BatchVectorizer(
            data_format='bow_n_wd', n_wd=n_dw_matrice.transpose(), vocabulary=self._idx2word,
            batch_size=self._config.get("batch_size", 1000)
        )
        print("BatchVectorizer initialised")

        model = LDA(
            num_topics=self._config["num_topics"], dictionary=bv.dictionary,
            num_document_passes=self._config.get("num_document_passes", 10),
            alpha=self._config.get("alpha", 0.01),
            beta=self._config.get("beta", 0.01),
            seed=self._config.get("seed", 2020))

        print(f"Fitting started")
        model.fit_offline(bv, self._config.get("num_collection_passes", 1))
        print(f"LDA model fitted: perplexity = {model.perplexity_value}, "
              f"sparsity phi = {model.sparsity_phi_value}, sparsity theta = {model.sparsity_theta_value}")
        self._model = model

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "model")
        self._model.save(model_path)
        save_with_pickle(self._config, path, "config")
        save_with_pickle(self._idx2word, path, "idx2word")

    @classmethod
    def load(cls, path: str):
        config = load_with_pickle(path, "config")
        idx2word = load_with_pickle(path, "idx2word")
        ret = cls(idx2word, config)
        lda = LDA(num_topics=config["num_topics"])
        lda.load(os.path.join(path, "model"))
        ret._model = lda

        return ret

    def get_top_tokens(self, num_tokens: int) -> List[List[str]]:
        return self._model.get_top_tokens(num_tokens)

    def get_token_topics(self, token: str) -> Optional[np.ndarray]:
        return self._model.phi_.loc[token, :].to_numpy()

    def get_documents_topics(self, n_dw_matrice) -> np.ndarray:
        bv = BatchVectorizer(
            data_format='bow_n_wd', n_wd=n_dw_matrice.transpose(), vocabulary=self._idx2word,
            batch_size=self._config.get("batch_size", 1000)
        )

        document_topics = self._model.transform(bv)
        return document_topics.to_numpy().T

    @property
    def num_topics(self):
        return self._config["num_topics"]


class TMDocumentVectorizer(AbstractVectorizer):
    def __init__(self, vectorizer: CountVectorizerWrapper, model: LDAModel):
        super().__init__()
        self._vectorizer = vectorizer
        self._model = model

    def __enter__(self):
        self._entered = True
        return self

    def _vectorize_doc(self, doc: Document) -> List[np.array]:
        raw_doc = {"tokens": doc.tokens}
        n_dw = self._vectorizer.transform([raw_doc])
        doc_topics = self._model.get_documents_topics(n_dw)[0]

        transformed_tokens = map(self._vectorizer.transform_token, doc.tokens)

        word_vectors = map(
            lambda t: np.zeros(self._model.num_topics) if t is None else self._model.get_token_topics(t),
            transformed_tokens)

        return list(map(lambda v: np.concatenate((doc_topics, v)), word_vectors))

    def __exit__(self, *exc):
        pass

    def save(self, path: str):
        save_with_pickle(self._vectorizer, path, "vectorizer")
        self._model.save(path)

    @classmethod
    def load(cls, path: str):
        return cls(load_with_pickle(path, "vectorizer"), LDAModel.load(path))
