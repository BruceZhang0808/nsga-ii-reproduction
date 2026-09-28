def dominate(p, q, objective_functions):
    """Return True if p dominates q else False"""
    flag = False    # To mark if there exits one function s.t. f(p) < f(q)
    for f in objective_functions:
        if f(p) > f(q):
            return False
        if f(p) < f(q):
            flag = True
    return True if flag else False


def fast_non_dominated_sort(population):
    N = len(population)
    n, S = [-1 for _ in range(N)], [[] for _ in range(N)]
    first_front = []
    for i in range(N):
        for j in range(N):
            if population[i] is population[j]:
                continue
            if dominate(population[i], population[j]):
                S[i].append(j)
            else:
                n[i] += 1
        if n[i] == 0:
            first_front.append(i)

    fronts = [first_front]
    i = 0
    while fronts[i]:
        next_front = []
        for p_index in fronts[i]:
            for q_index in S[p_index]:
                n[q_index] -= 1
                if n[q_index] == 0:
                    next_front.append(q_index)
        i += 1
        fronts.append(next_front)
    fronts.pop()

    return fronts