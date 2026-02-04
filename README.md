# Transformer LM (MLX)

A minimal decoder-only Transformer language model for learning and experimenting with small datasets using the MLX library.



## Quick overview

- Training and inference code lives in `main.py`.
- Dataset loading is in `datasets.py` (supports `ptb`, `wikitext2`, `wikitext103`, `enwik8`).
- Models and checkpoints are saved under the directory specified by `--save_dir` (default: `/Volumes/RAMDisk/transformer-lm`) as SHA1-named files with extensions `.safetensors` and `.metadata`. You can set `--save_dir` to any writable path.

## Requirements

See `requirements.txt` for Python package dependencies. Recommended Python version: **3.10+**.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install packages:

```bash
pip install -r requirements.txt
```

3. (Optional) Create the model/data directory used by default:

```bash
mkdir -p /Volumes/RAMDisk/transformer-lm
```

## Usage

Train (example — full run):

```bash
python main.py --dataset ptb --batch_size 32 --num_iters 100000 --save_dir /path/to/save
```

Quick smoke test (fast, for checks):

```bash
python main.py --dataset ptb --batch_size 2 --num_iters 10 --context_size 48 --new_model --save_dir ./tmp
```

Run interactive inference (CPU):

```bash
python main.py --inference
```

Run interactive inference (Metal GPU):

```bash
python main.py --inference --gpu
```

Start a new model from scratch (explicit):

```bash
python main.py --new_model
```

Resume from the latest checkpoint (default): the script will automatically pick the latest `.safetensors` file in `--save_dir`. You can also pass `--identifier <id>` to load a specific file.

Common flags and defaults:
- `--save_dir` : path to model/data directory (default: `/Volumes/RAMDisk/transformer-lm`)
- `--gpu` : enable Metal backend (if supported)
- `--seed` : RNG seed (default: 42)
- `--context_size` : context window size in tokens (default: 48)
- `--num_blocks` : number of Transformer blocks (default: 4)
- `--dim` : embedding/hidden dim (default: 1024)
- `--num_heads` : number of attention heads (default: 8)
- `--batch_size` : minibatch size (default: 32)
- `--num_iters` : training iterations (default: 100000)
- `--learning_rate` : AdamW learning rate (default: 3e-4)
- `--weight_decay` : weight decay (default: 1e-5)
- `--top_k` : top-k sampling (default: 50)
- `--temperature` : sampling temperature (default: 0.5)
- `--steps_per_report` / `--steps_per_eval` : monitoring frequency (defaults: 10 / 1000)
- `--identifier` : explicit model file identifier to load
- `--new_model` : start from scratch (ignore any existing checkpoint)
- `--inference` : start interactive generation mode
- `--checkpoint` : enable gradient checkpointing (memory-save option)

## Checkpoint & interrupts

- Checkpoints are saved atomically as `<sha1>.safetensors` and `<sha1>.metadata`.
- Single Ctrl+C: the program will finish the current job, save, then exit.
- Double Ctrl+C: force exit immediately (may abort save).

## Datasets

The project downloads and uses dataset files in the save directory (default: `/Volumes/RAMDisk/transformer-lm`). The `datasets.py` module will automatically download PTB and WikiText datasets if they are missing. This assumes you have a RAM disk mounted at `/Volumes/RAMDisk`; the `transformer-lm` directory will be created automatically if it does not already exist. If you do not use a RAM disk, set `save_dir` to a different path when calling `datasets.load_dataset` or create the directory manually.

## Notes

- The code relies on the `mlx` library (MLX core/nn/optimizers). If it is not available via pip in your environment, please install MLX per your local instructions or point pip at the appropriate wheel/source.
- Small test runs are recommended before long training runs. Use small `--num_iters` and reduced `--context_size` for debugging.

## Behavior & implementation notes

- Save and checkpointing:
  - Checkpoints are written atomically by first saving to `<sha1>.safetensors.new` and `<sha1>.metadata.new`, then renaming to `.safetensors` and `.metadata`. The metadata file contains the optimizer state, the dataset position (`s`), the current permutation (`perm`) and a `start` iteration used when resuming.
  - If an error occurs during save, the script retries; a double Ctrl+C forces an immediate exit and aborts save retries.
- Automatic resume: if `--identifier` is not provided, the script selects the latest `.safetensors` file from `--save_dir` and attempts to load it.
- Visualization: after each checkpoint (or when finishing), the script prints a short sample continuation (uses `--top_k` and `--temperature` for sampling).
- `--save_dir` note: datasets are downloaded into `--save_dir` and the `datasets` loader creates subdirectories as needed. If you do not want to use a RAM disk, set `--save_dir` to a path on your local disk and ensure it is writable.