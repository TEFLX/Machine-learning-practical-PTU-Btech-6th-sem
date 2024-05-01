import csv
import random
from sklearn.metrics import confusion_matrix, accuracy_score

print("___________Ritik kashyap__________")
def load_csv(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            dataset.append(row)
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        if row[column].strip().replace('.', '', 1).isdigit():
            row[column] = float(row[column].strip())

def split_dataset(dataset, split_ratio):
    train_size = int(len(dataset) * split_ratio)
    train_set = []
    test_set = list(dataset)
    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))
    return train_set, test_set
def separate_by_class(dataset):
    separated = {}
    for row in dataset:
        class_value = row[-1]
        if class_value not in separated:
            separated[class_value] = []
        separated[class_value].append(row)
    return separated
def frequency_table(dataset):
    freq_table = {}
    for class_value, instances in dataset.items():
        freq_table[class_value] = {}
        for instance in instances:
            for i in range(len(instance)-1):
                value = instance[i]
                if i not in freq_table[class_value]:
                    freq_table[class_value][i] = {}
                if value not in freq_table[class_value][i]:
                    freq_table[class_value][i][value] = 0
                freq_table[class_value][i][value] += 1
    return freq_table
def likelihood_table(freq_table):
    likelihoods = {}
    for class_value, columns in freq_table.items():
        likelihoods[class_value] = {}
        total_instances = sum(sum(column.values()) for column in columns.values())
        for column, value_freq in columns.items():
            likelihoods[class_value][column] = {}
            for value, freq in value_freq.items():
                likelihoods[class_value][column][value] = freq / total_instances
    return likelihoods
def class_probabilities(dataset):
    class_probs = {}
    total_instances = sum(len(instances) for instances in dataset.values())
    for class_value, instances in dataset.items():
        class_probs[class_value] = len(instances) / total_instances
    return class_probs
def calculate_posterior(instance, class_probs, likelihoods):
    posteriors = {}
    for class_value, class_prob in class_probs.items():
        posterior = class_prob
        for column, value in enumerate(instance):
            if column in likelihoods[class_value] and value in likelihoods[class_value][column]:
                posterior *= likelihoods[class_value][column][value]
        posteriors[class_value] = posterior
    return posteriors
def predict(class_probs, likelihoods, instance):
    posteriors = calculate_posterior(instance, class_probs, likelihoods)
    best_label, best_prob = None, -1
    for class_value, posterior in posteriors.items():
        if best_label is None or posterior > best_prob:
            best_prob = posterior
            best_label = class_value
    return best_label
def naive_bayes(train, test):
    separated = separate_by_class(train)
    freq_table = frequency_table(separated)
    likelihoods = likelihood_table(freq_table)
    class_probs = class_probabilities(separated)
    predictions = []
    for instance in test:
        result = predict(class_probs, likelihoods, instance)
        predictions.append(result)
    return predictions
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    true_labels = []
    predicted_labels = []
    folds = []
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = random.randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        folds.append(fold)
    for fold in folds:
        train_set = list(dataset_copy)
        for instance in fold:
            train_set.append(instance)
        test_set = []
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        true_labels.extend([row[-1] for row in fold])
        predicted_labels.extend(algorithm(train_set, test_set))
    return true_labels, predicted_labels
def test_naive_bayes(filename, split_ratio, n_folds):
    dataset = load_csv(filename)
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    str_column_to_float(dataset, len(dataset[0]) - 1)
    true_labels, predicted_labels = evaluate_algorithm(dataset, naive_bayes, n_folds, split_ratio)

    confusion = confusion_matrix(true_labels, predicted_labels)
    print('Confusion Matrix:')
    print(confusion)
    accuracy = accuracy_score(true_labels, predicted_labels)
    print('Accuracy: %.3f%%' % (accuracy * 100))

filename = 'tech_course_dataset.csv'  # Replace with your dataset file
split_ratio = 0.8  # 80% training data, 20% testing data
n_folds = 5  # 5-fold cross-validation
test_naive_bayes(filename, split_ratio, n_folds)
