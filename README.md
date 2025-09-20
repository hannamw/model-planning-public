# model-planning

This is the repository for the paper *Latent Planning Emerges with Scale*. In order to replicate the results reported in the paper, you will need a GPU with at least 40GB of VRAM. Perform the following steps:

1. Make sure that you've pulled `circuit-tracer` correctly as a submodule. If you forgot to pull the submodules too, you can use `git submodule update --init`
2. Create a virtual environment with all of the packages needed to run this paper. We recommend using [uv](https://docs.astral.sh/uv/), and have written this README assuming you use it. , just run `uv sync` in this directory. All of the python scripts that follow be run with `uv run [script]`.
3. Download all of the feature files by running `./download_features.sh`. Note that this assumes you are using `uv`, but just remove the `uv run` if not.
<!---
Create a directory `features/` and download each of the features for each of the publicly available `Qwen3` transcoders from [this collection](https://huggingface.co/collections/mwhanna/qwen-3-transcoders-68c3ed66393d1f86bff237a3), into a subdirectory with the same name as the model, e.g. `Qwen3-0.6B`. You can do this by running the command `huggingface-cli download [hf-repo-name] --include "features/*" --local-dir [model-name] --local-dir-use-symlinks False` within the `features` folder.
-->
4. Run the experiment scripts in each of the following subdirectories, which correspond to one set of our experiments.
    - `a_an`:
        1. `evaluate_professions.py`: gets behavioral results
        2. `compute_a_an_graphs.py`: computes graphs
        3. `planning_node_intervention.py`: computes all-effects, direct-effects, and random-node intervention effects for a_an examples.
        4. 
    - `is_are`:
    - `el_la`:
    - `couplets`:
5. The previous step suffices to replicate all of our work, but should you want to actually view the graphs created, you can run.