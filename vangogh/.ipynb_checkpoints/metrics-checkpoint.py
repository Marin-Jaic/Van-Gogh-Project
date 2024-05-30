import numpy as np
import pandas as pd

def scalability(model_configs, run_algo, cfg_kwargs=None):
    pops = []
    pts = []
    ts = [] 
    for config in model_configs:
        # pop_out = population_analysis(config, run_algo) if cfg_kwargs is None else population_analysis(config, run_algo, **cfg_kwargs)
        pop_out = np.zeros((2, 10))
        pop_out[0] = np.logspace(np.log10(1000), np.log10(1000000), num=10)
        pop_out[1] = np.random.randint(0, 1000, size=(10,))
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

def population_analysis(default_settings, run_algo, \
                            thresh_range=(50000, 500000), thresh_mode='exp', thresh_n=10, 
                            pop_range=(10, 10000), pop_mode='linear', pop_n=8):
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
            pop_step = pop_steps[pop_i]
            default_settings[1] = int(pop_step)
            
            data = run_algo(default_settings)
            df = pd.DataFrame(data)
            pop_fitness = np.min(df["best-fitness"])
            print(f"pop_fitness: <{pop_fitness}>, thresh: <{thresh}> ")
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

def time_analysis(default_settings, run_algo, \
                            thresh_range=(50000, 500000), thresh_mode='exp', thresh_n=10, 
                            population = 100):
    # Define threh range
    if thresh_mode == 'linear':
        thresh_steps = np.linspace(*thresh_range, thresh_n)
    elif thresh_mode == 'exp':
        thresh_steps = np.logspace(np.log10(thresh_range[0]), np.log10(thresh_range[1]), num=thresh_n)
        
    print("thresh_steps : ", thresh_steps)
    
    out = np.zeros(thresh_n)
    out[0] = thresh_steps
    for thresh_idx, thresh in enumerate(thresh_steps):
        pop_found = False
        # Set population in settings
        default_settings[1] = int(population)
        
        data = run_algo(default_settings)
        df = pd.DataFrame(data)
        pop_fitness = np.min(df["best-fitness"])
        print(f"pop_fitness: <{pop_fitness}>, thresh: <{thresh}> ")
        if pop_fitness < thresh:
            out[thresh_idx] = df["time-elapsed"]
        else:
            out[thresh_idx] = -1

    out_filename = "pop_analysis.npy"
    with open(out_filename, 'wb') as f:
        np.save(f, out)
    
    return out

def point_analysis(default_settings, run_algo, \
                            thresh_range=(50000, 500000), thresh_mode='exp', thresh_n=10, 
                            point_range=(10, 200), point_mode='linear', point_n=8):
    # Define threh range
    if thresh_mode == 'linear':
        thresh_steps = np.linspace(*thresh_range, thresh_n)
    elif thresh_mode == 'exp':
        thresh_steps = np.logspace(np.log10(thresh_range[0]), np.log10(thresh_range[1]), num=thresh_n)
        
    # Define pointulation range
    if point_mode == 'linear':
        point_steps = np.linspace(*point_range, point_n)
    elif point_mode == 'exp':
        point_steps = np.logspace(np.log10(point_range[0]), np.log10(point_range[1]), num=point_n)

    print("thresh_steps : ", thresh_steps)
    print("point_steps : ", point_steps)
    
    out = np.zeros((2, thresh_n))
    out[0] = thresh_steps
    for thresh_idx, thresh in enumerate(thresh_steps):
        point_found = False
        point_i = 0
        while not point_found and point_i < len(point_steps):
            # Set point in settings
            point_step = point_steps[point_i]
            default_settings[3] = int(point_step)
            
            data = run_algo(default_settings)
            df = pd.DataFrame(data)
            point_fitness = np.min(df["best-fitness"])
            print(f"point_fitness: <{point_fitness}>, thresh: <{thresh}> ")
            if point_fitness < thresh:
                out[1, thresh_idx] = point_step
                point_found = True
            point_i += 1
        if not point_found:
            out[1, thresh_idx] = -1

    out_filename = "point_analysis.npy"
    with open(out_filename, 'wb') as f:
        np.save(f, out)
    
    return out