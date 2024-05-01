import csv
print("________Ritik kashyap _________")
def find_s_algorithm(training_data):
    hypothesis = ['0', '0', '0', '0', '0', '0']

    print(f"\nInitial Hypothesis: {hypothesis}")
    for idx, example in enumerate(training_data):
        print(f"\nStep {idx + 1}:")
        print(f"Training Example No: {idx}")
        print(f"Current Hypothesis: {hypothesis}")

        # Check if the example is positive ('yes')
        if example[-1] == 'yes':
            for i in range(len(hypothesis)):
                if hypothesis[i] == '0':
                    hypothesis[i] = example[i]
                elif hypothesis[i] != example[i]:
                    hypothesis[i] = '?'

        print(f"Updated Hypothesis: {hypothesis}")

    return hypothesis

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

    initial_hypothesis = ['0', '0', '0', '0', '0', '0']
    print(f"\nThe initial value of hypothesis: {initial_hypothesis}")

    final_hypothesis = find_s_algorithm(training_data)
    print(f"\nThe Maximally Specific Hypothesis for the given Training Examples: {final_hypothesis}")

if __name__ == "__main__":
    main()
