import csv

from model import NaiveBayesClassifier, tokenize

dataset_file = input("evaluate model against dataset: (defaults to data/dataset_test.csv) ")
dataset_file = dataset_file if dataset_file != "" else "data/dataset_test.csv"

model_file = "data/model.json"

verbose = True

if __name__ == "__main__":
    dataset = []

    with open(dataset_file, newline = "") as load_file:
        load_csv = csv.reader(load_file)

        next(load_csv)
        for row in load_csv:
            dataset.append({
                "text": row[0],
                "label": row[1]
            })

    num_correct = 0
    num_incorrect = 0
    num_total = 0

    classifier = NaiveBayesClassifier.load_model(model_file)
    for data in dataset:
        predicted_label = classifier.classify(tokenize(data["text"]))
        if predicted_label == data["label"]:
            num_correct += 1
        else:
            num_incorrect += 1
            if verbose:
                print("incorrect: %s (%s) -> %s" % (data["text"], data["label"], predicted_label))

        num_total += 1

    print("correct: %d" % num_correct)
    print("incorrect: %d" % num_incorrect)
    print("total: %d" % num_total)

    print()

    accuracy = num_correct / num_total * 100.0
    print("accuracy: %.2f" % accuracy)
