# %%
from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from tqdm.notebook import tqdm
import numpy as np

from eval import get_run_metrics, read_run_dir, get_model_from_run
from plot_utils import basic_plot, collect_results, relevant_model_names

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')

run_dir = "../models"

# %%
df = read_run_dir(run_dir)
df  # list all the runs in our run_dir

# %%
task = "linear_regression"
#task = "sparse_linear_regression"
# task = "decision_tree"
#task = "relu_2nn_regression"

run_id = "pretrained_complete"  # if you train more models, replace with the run_id from the table above

run_path = os.path.join(run_dir, task, run_id)
recompute_metrics = False

if recompute_metrics:
    get_run_metrics(run_path)  # these are normally precomputed at the end of training

# %% [markdown]
# # Plot pre-computed metrics

# %%
def valid_row(r):
    return r.task == task and r.run_id == run_id

metrics = collect_results(run_dir, df, valid_row=valid_row)
_, conf = get_model_from_run(run_path, only_conf=True)
n_dims = conf.model.n_dims

models = relevant_model_names[task]
basic_plot(metrics["standard"], models=models)
plt.show()

# %% [markdown]
# # Interactive setup
# 
# We will now directly load the model and measure its in-context learning ability on a batch of random inputs. (In the paper we average over multiple such batches to obtain better estimates.)

# %%
from samplers import get_data_sampler
from tasks import get_task_sampler
print(conf.training.curriculum.points.end)

# %%
model, conf = get_model_from_run(run_path)

n_dims = conf.model.n_dims
batch_size = conf.training.batch_size
data_sampler = get_data_sampler(conf.training.data, n_dims)
task_sampler = get_task_sampler(
    conf.training.task,
    n_dims,
    batch_size,
    **conf.training.task_kwargs
)

task = task_sampler()
xs = data_sampler.sample_xs(b_size=batch_size, n_points=conf.training.curriculum.points.end)
ys = task.evaluate(xs)
with torch.no_grad():
    pred = model(xs, ys)
metric = task.get_metric()
loss = metric(pred, ys).numpy()

sparsity = conf.training.task_kwargs.sparsity if "sparsity" in conf.training.task_kwargs else None
baseline = {
    "linear_regression": n_dims,
    "sparse_linear_regression": sparsity,
    "relu_2nn_regression": n_dims,
    "decision_tree": 1,
}[conf.training.task]

plt.plot(loss.mean(axis=0), lw=2, label="Transformer")
plt.axhline(baseline, ls="--", color="gray", label="zero estimator")
plt.xlabel("# in-context examples")
plt.ylabel("squared error")
plt.legend()
plt.show()

# %% [markdown]
# ### 4.1 Sample Selection / Covariate Shifts

# %%
def R_Square_Error(ys, pred):
    # Step 2: Calculate the mean of the actual outcomes
    y_mean = torch.mean(ys)

    # Step 3: Compute SS_tot and SS_res
    SS_tot = torch.sum((ys - y_mean) ** 2)
    SS_res = torch.sum((ys - pred) ** 2)

    # Step 4: Calculate R^2
    R_square = 1 - SS_res / SS_tot
    return R_square

# %% [markdown]
# ####  Part 1 - Sample Selection: W1 = 1, w2 = 0

# %%
n_batches = 100
prompt_length = 2*conf.training.curriculum.points.end-1

# %%
actual_points_random = [[] for _ in range(prompt_length)]
predicted_points_random = [[] for _ in range(prompt_length)]

xs_list = torch.load('./data/xs_list.pth')
ys_list = torch.load('./data/ys_list.pth')
# Generate data and perform the experiment
for batch_idx in tqdm(range(n_batches)):
# for batch_idx in tqdm(range(1)):
    xs = xs_list[batch_idx]
    ys = ys_list[batch_idx]

    # j's idx starts at 1, but in graph, we refer it to 0
    pred = model(xs, ys)
    actual_points_random[0].extend(ys[:, 0])
    predicted_points_random[0].extend(pred[:, 0])
    
    for j in range(1, prompt_length):
        permuted_ys = np.copy(ys)
        if j > 2:  # if j == 1, 2, there are no prior labels or no need to permuted
            for i in range(batch_size):
                num_elements_to_shuffle = int(j * 0.10)
                indices_to_shuffle = np.random.choice(j, num_elements_to_shuffle, replace=False)
                elements_to_shuffle = permuted_ys[i, indices_to_shuffle]
                np.random.shuffle(elements_to_shuffle)
                permuted_ys[i, indices_to_shuffle] = elements_to_shuffle
            # print('sample', num_elements_to_shuffle)
                
        # Transfer np array to tensor
        permuted_ys_tensor = torch.from_numpy(permuted_ys)
        # predict with the si
        with torch.no_grad():
            pred = model(xs, permuted_ys_tensor)
        
        # for j in range(prompt_length):
        actual_points_random[j].extend(ys[:, j])
        predicted_points_random[j].extend(pred[:, j])

R_square_values_w_random_10p = []

for point_idx in range(prompt_length):
    actual = torch.tensor(actual_points_random[point_idx])
    predicted = torch.tensor(predicted_points_random[point_idx])
    R_square = R_Square_Error(actual, predicted)
    R_square_values_w_random_10p.append(R_square)

# %%

# %%
with open('./data/R_square_values_w_random_10p.txt', 'w') as f:
    for value in R_square_values_w_random_10p:
        f.write(f"{value}\n")
print("Finished R_square_values_w_random_10p")





