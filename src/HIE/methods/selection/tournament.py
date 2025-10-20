
import random

# def parent_selection(population, m):
#     tournament_size = 2
#     parents = []
#     while len(parents) < m:
#         tournament = random.sample(population, tournament_size)
#         tournament_fitness = [fit['objective'] for fit in tournament]
#         winner = tournament[tournament_fitness.index(min(tournament_fitness))]
#         parents.append(winner)
#     return parents


def parent_selection(population, m):
    tournament_size = 2
    parents = []
    indices = []

    while len(parents) < m:
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament = [population[i] for i in tournament_indices]
        tournament_fitness = [fit['objective'] for fit in tournament]
        winner_index = tournament_fitness.index(min(tournament_fitness))
        winner = tournament[winner_index]

        parents.append(winner)
        indices.append(tournament_indices[winner_index])

    return parents, indices