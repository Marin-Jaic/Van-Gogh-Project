import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading

def scalability(model_configs, run_algo, cfg_kwargs=None):
    pops = []
    pts = []
    ts = [] 
    for config in model_configs:
        pop_out = population_analysis(config, run_algo) if cfg_kwargs is None else population_analysis(config, run_algo, **cfg_kwargs)
        pts_out = pop_out # TODO set to point_analysis
        ts_out = pop_out # TODO set to time_analysis
        pops.append(pop_out)
        pts.append(pts_out)
        ts.append(ts_out)

    fig, axs = plt.subplots(1, 3)
    fig.set_figwidth(20)
    fig.set_figheight(5)
    plt_names = ["Population Analysis", "Point Analysis", "Time Analysis"]
    y_names = ["Population Size", "# of Points", "Time(s)"]
    y_lims = [[0, 1000], [0, 1000], [0, 1000]] # TODO Set this when other plots are done
    
    for i in range(3):
        axs[i].set_xscale('log')
        axs[i].set_yscale('log')
        axs[i].set_xlim([1000, 1000000])
        axs[i].set_ylim(np.array(y_lims[i]) + 1)
        axs[i].set_title(plt_names[i])
        axs[i].set_xlabel("MSE Threshold")
        axs[i].set_ylabel(y_names[i])

    # Plot output and save
    for i in range(len(model_configs)):
        axs[0].plot(pops[i][0], pops[i][1])
        axs[1].plot(pts[i][0], pts[i][1])
        axs[2].plot(ts[i][0], ts[i][1])
    plt.savefig('scalability_analysis.png')

# Tweak the thresh_range parameters, (maybe set to 10k)
# Make sure that the pop. size is % by 4
def population_analysis(default_settings, run_algo, \
                            thresh_range=(10000, 120000), thresh_mode='exp', thresh_n=2, 
                            pop_range=(100, 2000), pop_mode='linear', pop_n=4):
    # Define threh range
    if thresh_mode == 'linear':
        thresh_steps = np.linspace(*thresh_range, thresh_n)
    elif thresh_mode == 'exp':
        thresh_steps = np.logspace(np.log10(thresh_range[0]), np.log10(thresh_range[1]), num=thresh_n)
        
    # Define population range
    if pop_mode == 'linear':
        pop_steps = np.linspace(*pop_range, pop_n)

    elif pop_mode == 'exp':
        pop_steps = np.logspace(np.log10(pop_range[0]), np.log10(pop_range[1]), num=pop_n)

    print("thresh_steps : ", thresh_steps)
    print("pop_steps : ", pop_steps)
    
    out = np.zeros((2, thresh_n))
    out[0] = thresh_steps
    for thresh_idx, thresh in enumerate(thresh_steps):
        pop_found = False
        pop_i = 0
        while not pop_found and pop_i < len(pop_steps):
            # Set population in settings
            # Needed in order to ensure that the population size is divisible by 4
            pop_step = (pop_steps[pop_i] // 4 ) * 4
            default_settings[1] = int(pop_step)
            
            print(f'Default settings: {default_settings}')
            data = run_algo(default_settings)
            df = pd.DataFrame(data)
            pop_fitness = np.min(df["best-fitness"])
            # print(f"pop_fitness: <{pop_fitness}>, thresh: <{thresh}> ")
            if pop_fitness < thresh:
                out[1, thresh_idx] = pop_step
                pop_found = True
            pop_i += 1
        if not pop_found:
            out[1, thresh_idx] = -1

    out_filename = "pop_analysis.npy"
    with open(out_filename, 'wb') as f:
        np.save(f, out)
    
    return out

def run_experiment(model_configs, run_algo, num_runs=10, cfg_kwargs=None):
    final_results = {}
    
    def helper(index, config, results):
        res = run_algo(config)
        results[index] = [episode['best-fitness'] for episode in res]
    
    for i, config in enumerate(model_configs):
        results = {}
        final_results[i] = {
            "mean": {},
            "0.05": {},
            "0.95": {}
        }
        threads = []
        for j in range(num_runs):
            thread = threading.Thread(target=helper, args=(j, config, results))
            thread.start()
            threads.append(thread)

        for j, thread in enumerate(threads):
            thread.join()

        final_results[i]["mean"] = [np.mean([results[j][k] for j in range(num_runs)]) for k in range(min(list(map(len, results.values()))))]
        final_results[i]["0.05"] = [np.quantile([results[j][k] for j in range(num_runs)], 0.05) for k in range(min(list(map(len, results.values()))))]
        final_results[i]["0.95"] = [np.quantile([results[j][k] for j in range(num_runs)], 0.95) for k in range(min(list(map(len, results.values()))))]
    
    return final_results

def plot_results(results, titles):
    for index in results:
        plt.plot(results[index]["mean"], color='red')
        plt.fill_between(range(len(results[index]["mean"])), results[index]["mean"], results[index]["0.05"], alpha=0.2, color='red')
        plt.fill_between(range(len(results[index]["mean"])), results[index]["mean"], results[index]["0.95"], alpha=0.2, color='red')
        plt.title(titles[index])
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.savefig(titles[index].replace(' ', '_').lower() + ".png")
        plt.clf()
