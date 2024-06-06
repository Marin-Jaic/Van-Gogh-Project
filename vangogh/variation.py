import numpy as np
import random
from vangogh.util import NUM_VARIABLES_PER_POINT



def crossover(genes, method="ONE_POINT"):
    parents_1 = np.vstack((genes[:len(genes) // 2], genes[:len(genes) // 2]))
    parents_2 = np.vstack((genes[len(genes) // 2:], genes[len(genes) // 2:]))
    offspring = np.zeros(shape=genes.shape, dtype=int)

    if method == "ONE_POINT":
        crossover_points = np.random.randint(0, genes.shape[1], size=genes.shape[0])
        for i in range(len(genes)):
            offspring[i,:] = np.where(np.arange(genes.shape[1]) <= crossover_points[i], parents_1[i,:], parents_2[i,:])
    
    # Uniform Crossover
    elif method == "UNIFORM":
        for i in range(0, len(genes), 2):
            off_1, off_2 = uniform_crossover(parents_1[i,:], parents_2[i,:])
            offspring[i, :] = off_1
            offspring[i+1,:] = off_2
            
    elif method == "UNIFORM_DISTINCT":
        print("rrr")
        for i in range(0, len(genes), 2):
            off_1, off_2 = uniform_crossover_distinct(parents_1[i,:], parents_2[i,:])
            offspring[i, :] = off_1
            offspring[i+1,:] = off_2

    # k-points
    elif method[0].isdigit() and method[1:] == '_POINT':
        num_points = int(list(filter(str.isdigit, method))[0])
        length = len(parents_1)
        crossover_points = np.sort(np.random.randint(0, genes.shape[1], size=(genes.shape[0], num_points)))
        
        offspring = parents_1.copy()

        for i in range(0, len(genes), 2):
            for j in range(crossover_points.shape[1]):
                start = crossover_points[i,j]
                end = length
                if j + 1 < crossover_points.shape[1]:
                    end = crossover_points[i,j+1]
                if j % 2 == 0:
                    offspring[i, start:end] = parents_2[i, start:end]
                else:
                    offspring[i+1, start:end] = parents_2[i, start:end]
    elif method[0].isdigit() and method[1:] == '_POINT_DISTINCT':
        num_points = int(list(filter(str.isdigit, method))[0])
        length = len(parents_1)
        crossover_points = np.sort(np.random.randint(0, int(genes.shape[1] // NUM_VARIABLES_PER_POINT), size=(genes.shape[0], num_points)))
        crossover_points = crossover_points * NUM_VARIABLES_PER_POINT
        offspring = parents_1.copy()

        for i in range(0, len(genes), 2):
            for j in range(crossover_points.shape[1]):
                start = crossover_points[i,j]
                end = length
                if j + 1 < crossover_points.shape[1]:
                    end = crossover_points[i,j+1]
                if j % 2 == 0:
                    offspring[i, start:end] = parents_2[i, start:end]
                else:
                    offspring[i+1, start:end] = parents_2[i, start:end]
    #space
    elif method[0].isdigit() and method[1:] == '_SPATIAL':
        offspring = np.zeros(shape=genes.shape, dtype=int)
        for i in range(0,len(genes),2):

            #reshape as points
            parent_1_points = []
            parent_2_points = []
            num_points = int(len(genes[i]) / NUM_VARIABLES_PER_POINT)
            for r in range(num_points):
                p = r * NUM_VARIABLES_PER_POINT
                x, y, r, g, b = parents_1[i,p:p + NUM_VARIABLES_PER_POINT]
                parent_1_points.append((x , y,r,g,b))

                x, y, r, g, b = parents_2[i,p:p + NUM_VARIABLES_PER_POINT]
                parent_2_points.append((x , y,r,g,b))
            
            offspring_points_1, offspring_points_2 = split_parents(parent_1_points, parent_2_points, int(method[0]), mutate=True)

            #flatten
            offspring[i, :] = np.concatenate(offspring_points_1).ravel().tolist()
            offspring[i+1, :] = np.concatenate(offspring_points_2).ravel().tolist()
    elif method[0].isdigit() and method[1:] == '_SPATIAL_DISTINCT':
        offspring = np.zeros(shape=genes.shape, dtype=int)
        for i in range(0,len(genes),2):

            #reshape as points
            parent_1_points = []
            parent_2_points = []
            num_points = int(len(genes[i]) / NUM_VARIABLES_PER_POINT)
            for r in range(num_points):
                p = r * NUM_VARIABLES_PER_POINT
                x, y, r, g, b = parents_1[i,p:p + NUM_VARIABLES_PER_POINT]
                parent_1_points.append([x , y,r,g,b])

                x, y, r, g, b = parents_2[i,p:p + NUM_VARIABLES_PER_POINT]
                parent_2_points.append([x , y,r,g,b])
            
            offspring_points_1, offspring_points_2 = split_parents(parent_1_points, parent_2_points, int(method[0]))

            #flatten
            offspring[i, :] = np.concatenate(offspring_points_1).ravel().tolist()
            offspring[i+1, :] = np.concatenate(offspring_points_2).ravel().tolist()
    else:
        raise Exception("Unknown crossover method")

    return offspring


# sort parent points on axis and split in middle, give half of each parent to child
# axis alternates between x and y every further split
def split_parents(parent_1_points, parent_2_points, split,  mutate=False):
    
    length = len(parent_1_points)
    if split%2 == 0:
        parent_1_points.sort(key=lambda y : y[1])
        parent_2_points.sort(key=lambda y : y[1])
    else:
        parent_1_points.sort(key=lambda x : x[0])
        parent_2_points.sort(key=lambda x : x[0])

    # recursive base case
    if split == 1:
        if mutate:
            return merge_points(parent_1_points[:int(length//2)], parent_2_points[int(length//2):]), merge_points(parent_2_points[:int(length//2)], parent_1_points[int(length//2):])
        else:
            return np.concatenate([parent_1_points[:int(length//2)], parent_2_points[int(length//2):]]), np.concatenate([parent_2_points[:int(length//2)], parent_1_points[int(length//2):]])

    left_side_1, left_side_2 = split_parents(parent_1_points[:int(length//2)], parent_2_points[:int(length//2)], split-1)
    right_side_1, right_side_2 = split_parents(parent_1_points[int(length//2):], parent_2_points[int(length//2):], split-1)

    if mutate:
        return merge_points(left_side_1, right_side_1), merge_points(left_side_2, right_side_2)
    else:
        return np.concatenate([left_side_1, right_side_1]), np.concatenate([left_side_2, right_side_2])

def merge_points(points_1, points_2):
    rand_num = np.random.randint(0,NUM_VARIABLES_PER_POINT)
    left_point = points_1[-1]
    right_point  = points_2[0]
    new_left_point = np.concatenate([left_point[:rand_num], right_point[rand_num:]])
    new_right_point = np.concatenate([right_point[:rand_num], left_point[rand_num:]])
    return np.concatenate([points_1[:-1],[new_left_point],[new_right_point], points_2[1:]] )

def uniform_crossover(parent1, parent2, p=0.5):
    off_1, off_2 = parent1.copy(), parent2.copy()

    for i in range(len(parent1)):
        if np.random.uniform(0.0, 1.0) >= p:
            off_1[i], off_2[i] = off_2[i], off_1[i]

    return off_1, off_2

def uniform_crossover_distinct(parent1, parent2, p=0.5):
    off_1, off_2 = parent1.copy(), parent2.copy()

    for i in range(int(len(parent1)//NUM_VARIABLES_PER_POINT)):
        if np.random.uniform(0.0, 1.0) >= p:
            off_1[i:i+NUM_VARIABLES_PER_POINT], off_2[i:i+NUM_VARIABLES_PER_POINT] = off_2[i:i+NUM_VARIABLES_PER_POINT], off_1[i:i+NUM_VARIABLES_PER_POINT]

    return off_1, off_2

def mutate(genes, feature_intervals, mutation_probability=0.1, 
           num_features_mutation_strength=0.05, mutation_distribution="UNIFORM",
           std=1.0):
    mask_mut = np.random.choice([True, False], size=genes.shape,
                                p=[mutation_probability, 1 - mutation_probability])

    mutations = generate_plausible_mutations(genes, feature_intervals,
                                             num_features_mutation_strength,
                                             mutation_distribution, std=std)

    offspring = np.where(mask_mut, mutations, genes)

    return offspring

# Try with different variances over time
# Play around with the mutation strength
def generate_plausible_mutations(genes, feature_intervals,
                                 num_features_mutation_strength=0.25, 
                                 mutation_distribution="UNIFORM",
                                 std=1.0):
    mutations = np.zeros(shape=genes.shape)

    for i in range(genes.shape[1]):
        range_num = feature_intervals[i][1] - feature_intervals[i][0]
        low = -num_features_mutation_strength / 2
        high = +num_features_mutation_strength / 2

        if mutation_distribution == "UNIFORM":
            mutations[:, i] = range_num * np.random.uniform(low=low, high=high,
                                                        size=mutations.shape[0])
        elif mutation_distribution == "NORMAL":
            mutations[:, i] = range_num * np.random.normal(loc=genes[:,i], scale=std,
                                                        size=mutations.shape[0])
        else:
            raise Exception("Unknown mutation distribution")
        mutations[:, i] += genes[:, i]

        # Fix out-of-range
        mutations[:, i] = np.where(mutations[:, i] > feature_intervals[i][1],
                                   feature_intervals[i][1], mutations[:, i])
        
        mutations[:, i] = np.where(mutations[:, i] < feature_intervals[i][0],
                                   feature_intervals[i][0], mutations[:, i])

    mutations = mutations.astype(int)
    return mutations
