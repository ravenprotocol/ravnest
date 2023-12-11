import random

def initialize_population(nodes, population_size, max_clusters):
    return [[random.randint(0, max_clusters - 1) for _ in nodes] for _ in range(population_size)]

def calculate_fitness(individual, nodes, L, t):
    clusters = {}
    for index, cluster_id in enumerate(individual):
        clusters.setdefault(cluster_id, {'ram':0, 'speed':0})
        clusters[cluster_id]['ram'] += nodes[index]['ram']
        clusters[cluster_id]['speed'] += nodes[index]['ram']//nodes[index]['bandwidth']

    # print('clusters: ', clusters)

    penalty_L = 0
    for cluster_spec in list(clusters.values()):
        if cluster_spec['ram'] <= L:
            penalty_L += (L - cluster_spec['ram'])


    penalty_t = 0
    speeds = []
    for cluster_spec in list(clusters.values()):
        speeds.append(cluster_spec['speed'])

    penalty_t = max(speeds) - min(speeds)

    # print(penalty_L, penalty_t)

    penalty = 100 * penalty_L + penalty_t

    # print(penalty)

    return penalty

def select_parents(population, fitness):
    tournament_size = 5
    parents = []
    for _ in range(2):
        tournament = random.sample(list(zip(population, fitness)), tournament_size)
        tournament.sort(key=lambda x: x[1])
        parents.append(tournament[0][0])
    return parents

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    return parent1[:crossover_point] + parent2[crossover_point:], parent2[:crossover_point] + parent1[crossover_point:]

def mutate(individual, mutation_rate, max_clusters):
    return [random.randint(0, max_clusters - 1) if random.random() < mutation_rate else gene for gene in individual]

def decode_solution(solution, numbers):
    clusters = {}
    for num, cluster_id in zip(numbers, solution):
        clusters.setdefault(cluster_id, []).append(num)
    return clusters

def genetic_algorithm(nodes, L, t=0, population_size=200, max_clusters=5, generations=500, mutation_rate=0.01):    
    population = initialize_population(nodes, population_size, max_clusters)
    # print('initial_pop: ', population)
    best_individual = None
    best_fitness = 1000000
    for generation in range(generations):
        fitness = []
        for individual in population:            
            fit = calculate_fitness(individual, nodes, L, t)
            if fit < best_fitness:
                best_individual = individual  
                best_fitness = fit  
            fitness.append(fit)
        # if 0 in fitness:
        #     return decode_solution(population[fitness.index(0)], numbers)
        new_population = []
        for _ in range(population_size // 2):
            parents = select_parents(population, fitness)
            child1, child2 = crossover(*parents)
            new_population.extend([mutate(child1, mutation_rate, max_clusters), mutate(child2, mutation_rate, max_clusters)])
        population = new_population

    # print(best_individual)
    return decode_solution(best_individual, nodes)

    # return decode_solution(population[fitness.index(min(fitness))], numbers)

# Example usage
ram_variants = [2,4,8,16,32,64,128]
bandwidth_variants = [random.randint(1,40) for _ in range(10)]

num_nodes = 30
nodes = [{'ram':random.choice(ram_variants), 'bandwidth':random.choice(bandwidth_variants)} for _ in range(num_nodes)]
print('Pool: ', nodes)

L = 325

final_clusters = genetic_algorithm(nodes, L)
final_clusters = dict(sorted(final_clusters.items()))

# print('\nOptimal Clusters: ', final_clusters)
for cluster_id, cluster in final_clusters.items():
    print('\nCluster ',cluster_id, ': ', cluster)
    cluster_ram = 0
    cluster_speed = 0
    for machine in cluster:
        cluster_ram += machine['ram']
        cluster_speed += machine['ram'] // machine['bandwidth']

    print('\nCluster Ram: ', cluster_ram, '\nCluster Speed: ', cluster_speed)
    print('\n--------------------------------\n')
