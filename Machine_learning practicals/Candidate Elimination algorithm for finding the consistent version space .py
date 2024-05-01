import csv

print("________Ritik kashyap _________")

def candidate_elimination(training_data):
    specific_hypothesis = ['0', '0', '0', '0', '0', '0']
    general_hypothesis = [['?', '?', '?', '?', '?', '?']]

    print(f"\nInitial Specific Hypothesis: {specific_hypothesis}")
    print(f"Initial General Hypothesis: {general_hypothesis}")

    for idx, example in enumerate(training_data):
        print(f"\nStep {idx + 1}:")
        print(f"Training Example No: {idx}")
        print(f"Current Specific Hypothesis: {specific_hypothesis}")
        print(f"Current General Hypothesis: {general_hypothesis}")

        if example[-1] == 'yes':
            for i in range(len(specific_hypothesis)):
                if specific_hypothesis[i] == '0':
                    specific_hypothesis[i] = example[i]
                elif specific_hypothesis[i] != example[i]:
                    specific_hypothesis[i] = '?'

            general_hypothesis = [h for h in general_hypothesis if is_consistent(example, h)]

            general_hypothesis = [h for h in general_hypothesis if not is_more_general(specific_hypothesis, h)]

        else:
            new_general_hypotheses = []
            for h in general_hypothesis:
                if is_consistent(example, h):
                    new_h = [h[i] if specific_hypothesis[i] == h[i] else '?' for i in range(len(specific_hypothesis))]
                    if new_h not in general_hypothesis and new_h not in new_general_hypotheses:
                        new_general_hypotheses.append(new_h)

            general_hypothesis += new_general_hypotheses

    return specific_hypothesis, general_hypothesis

def is_consistent(instance, hypothesis):
    for i in range(len(hypothesis)):
        if hypothesis[i] != '0' and hypothesis[i] != instance[i]:
            return False
    return True
def is_more_general(hypothesis1, hypothesis2):
    for h1, h2 in zip(hypothesis1, hypothesis2):
        if h1 != h2 and h1 != '0':
            return False
    return True
def main():
    filename = 'training_data2.csv'
    training_data = []

    with open(filename, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            training_data.append(row)

    print("The Given Training Data Set:")
    for example in training_data:
        print(example)

    specific_hypothesis, general_hypothesis = candidate_elimination(training_data)
    print("\nThe Consistent Version Space:")
    print("Specific Hypothesis:", specific_hypothesis)
    print("General Hypothesis:", general_hypothesis)

if __name__ == "__main__":
    main()
