import json
import argparse
import multiprocessing

import nltk


def _segment_obj(obj: dict) -> dict:
    sentences = []
    tokens = []
    token_sent_start = 0

    for sent in nltk.sent_tokenize(obj["text"]):
        sent_tokens = nltk.word_tokenize(sent, preserve_line=True)
        sentences.append([token_sent_start, token_sent_start + len(sent_tokens)])
        token_sent_start += len(sent_tokens)
        tokens.extend(sent_tokens)

    return {**obj, "tokens": tokens, "sentences": sentences}


def main():
    parser = argparse.ArgumentParser(description='RCV nltk segmentor')
    parser.add_argument('input_json')
    parser.add_argument('output_path')
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)

    print(f"{len(raw_docs)} docs to be processed")

    with multiprocessing.Pool() as p:
        segmented = p.map(_segment_obj, raw_docs)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(segmented, f)


if __name__ == "__main__":
    main()
