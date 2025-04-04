"""Run a bunch of experiments in a loop"""
import os
import runpy
import sys
import time
import numpy as np

from tensorboard.backend.event_processing import event_accumulator

import wandb
from datetime import date

project_name = "finalfinal-cstr-sac-"+f"{date.today()}"

trials = 5
# batch_size = [64]
# policy_lr = [1e-4, 1e-3]
# q_lr = [1e-3]
critic_mode = ["mpcritic"]
scale = [1.0]
loss = [None]
H = [5]

experiment_num = 0
for seed in range(trials):
    for cm in critic_mode:
        for s in scale:
            for h in H:
                for l in loss:

                    experiment_num += 1
                    print("Experiment: ", experiment_num)
                    # print(f"Seed: {seed}", f"critic_mode: {cm}")

                    # the empty string in the first spot is actually important!
                    sys.argv = ["",  f"--seed={int(seed)}", f"--critic_mode={str(cm)}", f"--scale={s}", f"--loss={l}", f"--H={int(h)}",
                                f"--wandb_project_name={project_name}"]
                    # sys.argv = ["",  f"--seed={int(seed)}", f"--critic_mode={str(cm)}",
                    #             f"--wandb_project_name={project_name}"]
                    experiment = runpy.run_path(path_name="algos/sac_continuous_action_v2.py", run_name="__main__")

