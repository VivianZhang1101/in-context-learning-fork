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
n_batches = 1 # We use n_batches = 1 here to speed up the experiment speed. We used n_batches = 100 for the experiment result in the paper
prompt_length = 2*conf.training.curriculum.points.end-1
# # save xs and ys for the 4.2 experiment
xs_list = [] 
ys_list = [] 
actual_points_w_1 = [[] for _ in range(prompt_length)]
predicted_points_w_1 = [[] for _ in range(prompt_length)]
i = 0
# Generate data and perform the experiment
for _ in tqdm(range(n_batches)):
    i += 1
    # if i%20 == 0:
    #     print(f"batch: {i}")
    xs = data_sampler.sample_xs(b_size=batch_size, n_points=prompt_length)
    # 64 x 101 x 20
    xs_list.append(xs)
    ys = task.evaluate(xs)
    # 64 x 101
    ys_list.append(ys)

    with torch.no_grad():
        pred = model(xs, ys)
    for j in range(prompt_length):
        actual_points_w_1[j].extend(ys[:, j])
        predicted_points_w_1[j].extend(pred[:, j])

R_square_values_w_1_root_2 = []

for point_idx in range(prompt_length):
    actual = torch.tensor(actual_points_w_1[point_idx])
    predicted = torch.tensor(predicted_points_w_1[point_idx])
    R_square = R_Square_Error(actual, predicted)
    R_square_values_w_1_root_2.append(R_square)
    
torch.save(xs_list, './data/xs_list.pth')
torch.save(ys_list, './data/ys_list.pth')


# %%
with open('./data/R_square_values_w_1_root_2.txt', 'w') as f:
    for value in R_square_values_w_1_root_2:
        f.write(f"{value}\n")
print("Finished R_square_values_w_1_root_2")

# %% [markdown]