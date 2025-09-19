# model-planning

This is the repository for the paper *Latent Planning Emerges with Scale*. In order to replicate the results reported in the paper, you will need a GPU with at least 40GB of VRAM. Perform the following steps:

1. Make sure that you've pulled `circuit-tracer` correctly as a submodule. If you forgot to pull the submodules too, you can use `git submodule update --init`
2. Create a virtual environment with all of the packages needed to run this paper. We recommend using [uv](https://docs.astral.sh/uv/); in this case, just run `uv sync` in this directory, and prepend all following commands with `uv run`.
3. Create a directory `features/` and download each of the features for each of the publicly available `Qwen3` transcoders from [this collection](https://huggingface.co/collections/mwhanna/qwen-3-transcoders-68c3ed66393d1f86bff237a3), into a subdirectory with the same name as the model, e.g. `Qwen3-0.6B`. You can do this by running the command `huggingface-cli download [hf-repo-name] --include "features/*" --local-dir [model-name] --local-dir-use-symlinks False` within the `features` folder.
4. Run all of the experiment scripts in each of the following subdirectories, which correspond to one set of our experiments.
    - `a_an`:
        1. `evaluate_professions.py`
        2. `
    - `is_are`:
    - `el_la`:
    - `couplets`:
5. The previous step suffices to replicate all of our work, but should you want to actually view the graphs created, you can run.