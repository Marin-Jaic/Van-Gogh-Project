from PIL import Image
from vangogh.evolution import Evolution
from vangogh.fitness import draw_voronoi_image
from vangogh.util import IMAGE_SHRINK_SCALE, REFERENCE_IMAGE
from IPython.display import display, clear_output
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
from mpl_toolkits.axes_grid1 import ImageGrid
plt.style.use('classic')
# %matplotlib inline

from multiprocess import Pool, cpu_count
from time import time
from vangogh.evolution import Evolution
from vangogh.metrics import run_experiment, plot_results

NUM_VARIABLES_PER_POINT = 5
IMAGE_SHRINK_SCALE = 6

REFERENCE_IMAGE = Image.open("./img/reference_image_resized.jpg").convert('RGB')

# display(REFERENCE_IMAGE)

# Enable to show live rendering of best individual during optimization
display_output = False
# Enable to save progress images at every 50th generation
save_progress = True
# Enable to print verbose output per generation
verbose_output = False

def reporter(time, evo):
    if save_progress or display_output:
        elite = draw_voronoi_image(evo.elite, evo.reference_image.width, evo.reference_image.height, scale=IMAGE_SHRINK_SCALE)
    if display_output:
        clear_output()
        display(elite)
    if save_progress and time["num-generations"] % 50 == 0:
        elite.save(f"./img/van_gogh_intermediate_{evo.seed}_{evo.population_size}_{evo.crossover_method}_{evo.num_points}_{evo.initialization}_{evo.generation_budget}_{time['num-generations']:05d}.png")

def run_algorithm(settings):
    seed, population_size, crossover_method, num_points, initialization, generation_budget, mutation = settings
    start = time()
    
    data = []
    evo = Evolution(num_points,
                    REFERENCE_IMAGE,
                    population_size=population_size,
                    generation_reporter=reporter,
                    crossover_method=crossover_method,
                    seed=seed,
                    initialization=initialization,
                    generation_budget=generation_budget,
                    num_features_mutation_strength=.25,
                    selection_name='tournament_4',
                    mutation_type=mutation,
                    verbose=verbose_output)
    data = evo.run()
    time_spent = time() - start
    # print(f"Done: run {seed} - pop {population_size} - crossover {crossover_method} - num. points {num_points} - initialization {initialization} - in {int(time_spent)} seconds")
    
    return data

if __name__ == "__main__":
    crossover_methods = ["ONE_POINT", "2_POINT", "5_POINT", "2_SPATIAL", "5_SPATIAL", "UNIFORM"]
    # crossover_methods = ["ONE_POINT", "2_POINT"]  # Dummy for testing
    mutation_type = "UNIFORM"
    configs = [[0, 100, method, 100, 'RANDOM', 500, mutation_type] for method in crossover_methods]

    results = run_experiment(configs, run_algorithm, num_runs=10)
    plot_results(results=results, titles=[f"Run with {method} {mutation_type}" for method in crossover_methods])
    print('Finished.')