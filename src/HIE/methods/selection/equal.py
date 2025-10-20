import random
# def parent_selection(population, m):
#     parents = random.choices(population, k=m)
#     return parents

def parent_selection(population, m):
    indices = random.choices(range(len(population)), k=m)
    parents = [population[i] for i in indices]
    return parents, indices