import numpy as np
print("________Ritik kashyap _________")
# Define your hyperparameters and their ranges
hyperparameters = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128],
    'num_hidden_units': [64, 128, 256]
}


# Define fitness function (evaluation metric)
def evaluate_hyperparameters(hyperparameters):
    # Run your machine learning model with the given hyperparameters and return the evaluation metric
    return np.random.random()


# Initialize population
population_size = 10
population = [{} for _ in range(population_size)]
for i in range(population_size):
    for param in hyperparameters:
        population[i][param] = np.random.choice(hyperparameters[param])

# Main loop
generations = 10
for gen in range(generations):
    # Evaluate fitness of the population
    fitness_scores = [evaluate_hyperparameters(individual) for individual in population]

    # Parent selection (select individuals for mating)
    parents_indices = np.random.choice(population_size, size=2, replace=False)
    parent1, parent2 = population[parents_indices[0]], population[parents_indices[1]]

    # Crossover with probability pc
    crossover_point = np.random.randint(1, len(hyperparameters))
    child = {**parent1, **parent2}

    # Mutation with probability pm
    mutation_param = np.random.choice(list(hyperparameters.keys()))
    child[mutation_param] = np.random.choice(hyperparameters[mutation_param])

    # Replace least fit individual with the child
    least_fit_index = np.argmin(fitness_scores)
    population[least_fit_index] = child

# Find best hyperparameters
best_hyperparameters = population[np.argmax(fitness_scores)]
best_fitness = evaluate_hyperparameters(best_hyperparameters)

print("Best Hyperparameters:", best_hyperparameters)
print("Best Fitness:", best_fitness)
