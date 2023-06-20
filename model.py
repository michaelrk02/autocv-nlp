import json
import math
import random
import re

import numpy as np

from nltk.grammar import CFG, Nonterminal, Production
from nltk.parse.earleychart import EarleyChartParser
from nltk.parse.generate import generate
from nltk.tree import Tree

from numpy.linalg import norm

from util import readfile

class Recognizer:
    def __init__(self, grammar_file, identifier = "STMT", variable = "VAR"):
        self.grammar = CFG.fromstring(readfile(grammar_file))
        self.variable = variable
        self.identifier = identifier

    def recognize(self, tokens):
        terminals = set()
        for p in self.grammar.productions():
            for s in p.rhs():
                if type(s) is str:
                    terminals.add(s)

        unknown = set()
        for t in tokens:
            if t not in terminals:
                unknown.add(t)

        additional_productions = []
        for t in unknown:
            additional_productions.append(Production(Nonterminal(self.variable), [t]))

        grammar = CFG(Nonterminal(self.identifier), self.grammar.productions() + additional_productions)
        parser = EarleyChartParser(grammar)
        for t in parser.parse(tokens):
            return t

        return None

class SemanticAnalyzer:
    def __init__(self, syntax_tree):
        self.syntax_tree = syntax_tree if syntax_tree is not None else Tree("", [])

    def tree(self):
        return self.syntax_tree

    def label(self):
        return self.syntax_tree.label()

    def get(self):
        return SemanticAnalyzer(self.syntax_tree[0])

    def get_all(self):
        result = []
        for t in self.syntax_tree:
            result.append(SemanticAnalyzer(t))
        return result

    def find(self, label):
        result = None

        for pos in self.syntax_tree.treepositions():
            t = self.syntax_tree[pos]
            if type(t) is Tree and t.label() == label:
                result = t
                break

        return SemanticAnalyzer(result)

    def find_all(self, label):
        result = []

        for pos in self.syntax_tree.treepositions():
            t = self.syntax_tree[pos]
            if type(t) is Tree and t.label() == label:
                result.append(SemanticAnalyzer(t))

        return result

    def text(self):
        return " ".join(self.syntax_tree.leaves())

    def test(self, label):
        return self.find(label).text() != ""

class Generator:
    def __init__(self, grammar_file, identifier = "STMT", depth = None):
        self.grammar = CFG.fromstring(readfile(grammar_file))
        self.identifier = identifier
        self.depth = depth

    def generate(self, semantic, return_iter = False, override = False):
        productions = []
        if not override:
            productions = self.grammar.productions()
        else:
            override_nonterminals = set(semantic.variables.keys())
            for p in self.grammar.productions():
                if p.lhs().symbol() not in override_nonterminals:
                    productions.append(p)

        additional_productions = []
        for k, words in semantic.variables.items():
            for w in words:
                additional_productions.append(Production(Nonterminal(k), [w]))

        grammar = CFG(Nonterminal(self.identifier), productions + additional_productions)

        if return_iter:
            return generate(grammar, depth = self.depth)

        candidate_count = 0
        for t in generate(grammar, depth = self.depth):
            if semantic.validate(t):
                candidate_count += 1

        candidate_index = 0
        candidate_choose = random.randint(0, candidate_count - 1)
        for t in generate(grammar, depth = self.depth):
            if semantic.validate(t):
                if candidate_index == candidate_choose:
                    return " ".join(t)

                candidate_index += 1

        return ""

class SemanticConstructor:
    def __init__(self, template = None):
        self.variables = template if not template is None else {}
        self._update_keywords()

    def set(self, variable, values):
        self.variables[variable] = values
        self._update_keywords()

    def validate(self, sentence):
        if type(sentence) is list:
            sentence = " ".join(sentence)

        is_complete = True
        for k, words in self.keywords.items():
            exists = False
            for w in words:
                if w in sentence:
                    exists = True
                    break

            if not exists:
                is_complete = False
                break

        return is_complete

    def _update_keywords(self):
        keywords = {}
        for k, words in self.variables.items():
            keywords[k] = set()
            for w in words:
                keywords[k].add(w)

        self.keywords = keywords

class NaiveBayesClassifier:
    def __init__(self, model):
        self.model = model

    def classify(self, tokens, result = "chosen"):
        likelihood = self.model["likelihood"]
        prior = self.model["prior"]
        vocab = self.model["vocab"]
        labels = self.model["labels"]

        sentence_vocab = set(tokens) & set(vocab)

        predictions = {}
        for l in labels:
            predictions[l] = math.log(prior[l])
            for v in sentence_vocab:
                predictions[l] += math.log(likelihood[v][l])

        min_value = math.inf
        max_value = -math.inf
        for v in predictions.values():
            if v < min_value:
                min_value = v
            if v > max_value:
                max_value = v

        magnitude = 0.0
        for k in predictions.keys():
            predictions[k] = (predictions[k] - min_value) / (max_value - min_value)
            magnitude += predictions[k]

        for k in predictions.keys():
            predictions[k] = predictions[k] / magnitude

        chosen_label = labels[0]
        for l in labels:
            if predictions[l] > predictions[chosen_label]:
                chosen_label = l

        if result == "chosen":
            return chosen_label
        elif result == "chosen_probability":
            return (chosen_label, predictions[chosen_label])
        elif result == "probabilities":
            return predictions

        return None

    def save_model(self, filename):
        with open(filename, "w") as model_file:
            json.dump(self.model, model_file)

    @classmethod
    def load_model(self, filename):
        model = None
        with open(filename) as model_file:
            model = json.load(model_file)

        return NaiveBayesClassifier(model)

    @classmethod
    def train_model(self, dataset, vocab, labels):
        label_vocab = {}
        for l in labels:
            label_vocab[l] = 0
            for data in dataset:
                if data["@label"] == l:
                    for v in vocab:
                        label_vocab[l] += data[v]

        likelihood = {}
        for v in vocab:
            likelihood[v] = {}

            for l in labels:
                frequency = 0
                for data in dataset:
                    if data["@label"] == l:
                        frequency += data[v]

                likelihood[v][l] = (frequency + 1) / (label_vocab[l] + len(vocab))

        prior = {}
        for l in labels:
            prior[l] = 0
            for data in dataset:
                if data["@label"] == l:
                    prior[l] += 1

            prior[l] /= len(data)

        model = {"likelihood": likelihood, "prior": prior, "vocab": vocab, "labels": labels}

        return NaiveBayesClassifier(model)

def tokenize(sentence):
    return [s.lower() for s in re.findall(r"[^\s.:,]+|[\.:,]", sentence)]

def chunk(tokens):
    segments = []
    segment = []
    for t in tokens:
        if t == ".":
            segments.append(segment)
            segment = []
        else:
            segment.append(t)

    return segments
