import random
import csv

from model import Generator, NaiveBayesClassifier, SemanticConstructor, tokenize
from util import nrandom

def generate_dataset(label, generator, semantic_constructor, sampling_frequency = 0.5):
    dataset = []
    dataset_iter = generator.generate(semantic_constructor, True, True)
    for data in dataset_iter:
        if semantic_constructor.validate(data):
            if nrandom() < sampling_frequency:
                dataset.append({"text": " ".join(data), "label": label})

    print(" generate(%s) -> %d" % (label, len(dataset)))

    return dataset

def get_dataset_vocab(dataset):
    vocab = set()
    for data in dataset:
        for t in tokenize(data["text"]):
            vocab.add(t)

    return vocab

def analyze_dataset(label, dataset):
    vocab = get_dataset_vocab(dataset)

    print("### Dataset: %s ###" % label)
    print(" length: %d" % len(dataset))
    print(" vocab length: %d" % len(vocab))

def initialize_dataset(store_filename, recognizers):
    dataset = []

    for r in recognizers:
        if r["enable"]:
            generator = Generator("grammars/%s.cfg" % r["label"], "STMT", r["depth"])
            semantic = SemanticConstructor(r["semantic"])

            dataset += generate_dataset(r["label"], generator, semantic, r["frequency"])

    dataset_vocab = get_dataset_vocab(dataset)

    with open(store_filename, "w", newline = "") as store_file:
        store_csv = csv.writer(store_file)

        store_csv.writerow(["text", "label"])
        for data in dataset:
            store_csv.writerow([data["text"], data["label"]])

def transform_dataset(load_filename, store_filename):
    dataset = []

    with open(load_filename, newline = "") as load_file:
        load_csv = csv.reader(load_file)

        next(load_csv)
        for row in load_csv:
            dataset.append({
                "text": row[0],
                "label": row[1]
            })

    vocab = get_dataset_vocab(dataset)

    transform = []
    for data in dataset:
        t = {}
        t["@label"] = data["label"]
        for v in vocab:
            t[v] = 1 if v in tokenize(data["text"]) else 0

        transform.append(t)

    with open(store_filename, "w", newline = "") as store_file:
        store_csv = csv.writer(store_file)

        header = []
        for v in vocab:
            header.append(v)

        header.append("@label")

        store_csv.writerow(header)

        for t in transform:
            row = []
            for v in vocab:
                row.append(t[v])

            row.append(t["@label"])

            store_csv.writerow(row)

def train_dataset(load_filename, save_filename):
    dataset = []
    dataset_vocab = []
    dataset_labels = []

    with open(load_filename, newline = "") as load_file:
        load_csv = csv.reader(load_file)

        header = next(load_csv)
        for h in header:
            if h[0] != "@":
                dataset_vocab.append(h)

        for row in load_csv:
            data = {}
            for i in range(len(dataset_vocab)):
                data[dataset_vocab[i]] = int(row[i])

            label = row[len(dataset_vocab)]
            if label not in dataset_labels:
                dataset_labels.append(label)

            data["@label"] = label
            dataset.append(data)

    classifier = NaiveBayesClassifier.train_model(dataset, dataset_vocab, dataset_labels)
    classifier.save_model(save_filename)

