# %%
from collections import OrderedDict
import re
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
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


# %% [markdown]
# # Interactive setup
# 
# We will now directly load the model and measure its in-context learning ability on a batch of random inputs. (In the paper we average over multiple such batches to obtain better estimates.)

# %%
from samplers import get_data_sampler
from tasks import get_task_sampler


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
# ####  Expriment 1
# %%
# ####  Part 1 - Sample Selection: W1 = 1, w2 = 0
'''
The following is a framwork for prompt tuning with one batch for t=10
'''


# Freeze all the model parameters
for param in model.parameters():
    param.requires_grad = False
    

prompt_length = 75

def backward_hook_xs(grad):
    grad[:, -1, :] = 0  # Set gradient for the last item to 0
    return grad

def backward_hook_ys(grad):
    grad[:, -1] = 0  # Set gradient for the last item to 0
    return grad

loss_list = []
for j in range(2, prompt_length+2):

    xs = data_sampler.sample_xs(b_size=batch_size, n_points=j) 
    ys = task.evaluate(xs)
    xs = nn.Parameter(xs)
    ys = nn.Parameter(ys)

    xs.requires_grad_(True)
    ys.requires_grad_(True)

    xs.register_hook(backward_hook_xs)
    ys.register_hook(backward_hook_ys)
    
    optimizer = torch.optim.SGD([xs, ys], lr=0.01)
    loss_fn = nn.MSELoss()
    N = 1500
    for step in range(N):
        optimizer.zero_grad() 
        # output with prompts xs and ys
        output = model(xs, ys) 
        xs_test = xs[:,-1,:].detach().unsqueeze(1)
        ys_test = ys[:, -1].detach().unsqueeze(1)
        # output only with xs[:,-1,:], this output acts as the labels to calculate loss
        test_target = model(xs_test, ys_test)
        output_target = output[:,-1].unsqueeze(1)
        loss = loss_fn(output_target, test_target)

        if step % 500 == 0:
            print(f'j={j} Step {step}: Loss = {loss.item()}')
        loss.backward()               
        optimizer.step()
        
    loss_list.append(loss)
        
    with open('./data/loss_list.txt', 'w') as f:
        for value in loss_list:
            f.write(f"{value}\n")
    print("Finished loss_list")

# %%



