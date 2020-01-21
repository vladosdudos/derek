import json
import argparse
import os
from datetime import datetime

from typing import List

from tools.common.helper import get_next_props
from topic_modelling.model import LDAModel, TMVectorizer, VectorizerWrapper


def _parse_possible_props(possible_props: List[dict]):
    vectorizer_props = []
    model_props = []

    for pr in possible_props:
        if pr["vectorizer"] not in vectorizer_props:
            vectorizer_props.append(pr["vectorizer"])
        if pr["model"] not in model_props:
            model_props.append(pr["model"])

    return vectorizer_props, model_props


def _compute_vocab_freq(n_dw, vocab):
    n_w = n_dw.sum(axis=0)
    vocab_freqs = {k: n_w[0, v] for k, v in vocab.items()}
    return vocab_freqs


def _dump_vocab_freqs(n_dw, vocab, path):
    vocab_freqs = _compute_vocab_freq(n_dw, vocab)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"{k}: {v}" for k, v in sorted(vocab_freqs.items(), key=lambda x: -x[1])))


def main():
    parser = argparse.ArgumentParser(description='RCV topic modelling fitter')
    parser.add_argument('input_json')
    parser.add_argument('props_json')
    parser.add_argument('lst_json')
    parser.add_argument('output_path')
    parser.add_argument("-top_tokens", default=0, type=int)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)

    print(f"Corpus size = {len(raw_docs)}")

    with open(args.props_json, "r", encoding="utf-8") as f:
        props = json.load(f)

    with open(args.lst_json, "r", encoding="utf-8") as f:
        lst = json.load(f)

    possible_props = list(get_next_props(props, lst))
    vectorizer_props, model_props = _parse_possible_props(possible_props)

    for v_idx, concrete_vect_props in enumerate(vectorizer_props):
        print("-" * 40)
        print(datetime.now())
        print(f"Vectorizer props {v_idx}: {concrete_vect_props}")
        vectorizer = TMVectorizer(concrete_vect_props)
        n_dw = vectorizer.fit_transform(raw_docs)
        vocab = vectorizer.get_vocab()
        print(f"Vocab size = {len(vocab)}")

        vect_path = os.path.join(args.output_path, f"vect_props_{v_idx}")
        os.makedirs(vect_path, exist_ok=True)

        _dump_vocab_freqs(n_dw, vocab, os.path.join(vect_path, "vocab_freqs.txt"))
        with open(os.path.join(vect_path, 'vect_props.json'), 'w', encoding="utf-8") as f:
            f.write(json.dumps(concrete_vect_props, indent=4, sort_keys=True))

        for m_idx, concrete_model_props in enumerate(model_props):
            print("=" * 40)
            print(datetime.now())
            print(f"Model props {m_idx}: {concrete_model_props}")

            model_path = os.path.join(vect_path, f"model_props_{m_idx}")
            os.makedirs(model_path, exist_ok=True)
            with open(os.path.join(model_path, 'model_props.json'), 'w', encoding="utf-8") as f:
                f.write(json.dumps(concrete_model_props, indent=4, sort_keys=True))

            model = LDAModel({v: k for k, v in vocab.items()}, concrete_model_props)
            model.fit(n_dw)

            wrapped = VectorizerWrapper(vectorizer, model)
            wrapped.save(os.path.join(model_path, "saved_model"))

            if args.top_tokens <= 0:
                continue

            with open(os.path.join(model_path, 'top_tokens.log'), "w", encoding="utf-8") as f:
                for j, topic_top in enumerate(model.get_top_tokens(args.top_tokens)):
                    print(f"topic {j}:" + ", ".join(topic_top), file=f)


if __name__ == "__main__":
    main()
