import random
# def parent_selection(pop,m):
#     ranks = [i for i in range(len(pop))]
#     probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
#     parents = random.choices(pop, weights=probs, k=m)
#     return parents

def parent_selection(pop, m):
    ranks = [i for i in range(len(pop))]
    probs = [1 / (rank + 1 + len(pop)) for rank in ranks]
    indices = random.choices(ranks, weights=probs, k=m)
    parents = [pop[i] for i in indices]
    return parents, indices