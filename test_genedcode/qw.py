import numpy as np


def algo(initial_population, individual_upper, individual_lower, objective_function):
    # Hybrid PSO-GA algorithm parameters
    population_size = len(initial_population)
    num_generations = 100
    inertia_weight = 0.7
    cognitive_coefficient = 1.5
    social_coefficient = 1.5

    # GA parameters
    mutation_rate = 0.05
    crossover_rate = 0.8

    # Initialize PSO particles
    particles = initial_population
    velocities = np.zeros_like(particles)

    # Main loop
    for gen in range(num_generations):
        # Evaluate fitness of current population
        fitness_values = [objective_function(sol) for sol in particles]

        # Update velocities based on PSO rules
        for i in range(population_size):
            r1 = np.random.rand()
            r2 = np.random.rand()
            velocities[i] = inertia_weight * velocities[i] + \
                            cognitive_coefficient * r1 * (particles[i] - particles[i]) + \
                            social_coefficient * r2 * (np.mean(particles, axis=0) - particles[i])

            # Ensure velocity stays within bounds
            velocities[i] = np.clip(velocities[i], -1, 1)

            # Update positions based on velocities
            particles[i] += velocities[i]

            # Ensure position stays within bounds
            particles[i] = np.clip(particles[i], individual_lower, individual_upper)

        # Apply GA operators (mutation and crossover)
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                # Mutate individual
                particles[i] = mutate_individual(particles[i])

            if np.random.rand() < crossover_rate:
                # Perform crossover with another individual
                particles[i], _ = crossover_individuals(particles[i], particles[np.random.randint(population_size)])

        # Evaluate fitness again after applying GA operators
        fitness_values = [objective_function(sol) for sol in particles]

        # Select best individuals for next generation
        sorted_indices = np.argsort(fitness_values)
        best_individuals = particles[sorted_indices[:population_size // 2]]

        # Replace worst half of population with best half
        particles = np.concatenate((best_individuals, particles[sorted_indices[population_size // 2:]]))

    # Return best solution found
    return particles[-1]