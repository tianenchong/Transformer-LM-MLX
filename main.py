from collections import deque
import hashlib
import itertools
import math
import os
import signal
import sys
import threading
import time
from functools import partial
from typing import Dict, Optional
import dill
import numpy as np
import datasets
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from safetensors import safe_open, numpy
import argparse

parser = argparse.ArgumentParser("Train a decoder-only Transformer LM with MLX.")
parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the RNGs.")
parser.add_argument(
    "--dataset",
    type=str,
    default="ptb",
    choices=["enwik8", "ptb", "wikitext2", "wikitext103"],
    help="Dataset to train and evaluate on.",
)
parser.add_argument(
    "--save_dir",
    type=str,
    default="/Volumes/RAMDisk/transformer-lm",
    help="Model and data directory for saving, default to /Volumes/RAMDisk/transformer-lm.",
)
parser.add_argument(
    "--context_size",
    type=int,
    default=48,
    help="Context size in tokens of the model.",
)
parser.add_argument(
    "--num_blocks", type=int, default=4, help="Number of Transformer blocks."
)
parser.add_argument(
    "--dim",
    type=int,
    default=1024,
    help="Dimensionality of embeddings and hidden layers.",
)
parser.add_argument(
    "--num_heads",
    type=int,
    default=8,
    help="Number of heads used for multi-head attention",
)
parser.add_argument(
    "--checkpoint", action="store_true", help="Perform gradient checkpointing"
)
parser.add_argument("--batch_size", type=int, default=32, help="Minibatch size.")
parser.add_argument(
    "--num_iters", type=int, default=100000, help="Iterations to train for."
)
parser.add_argument(
    "--learning_rate", type=float, default=3e-4, help="AdamW learning rate."
)
parser.add_argument(
    "--weight_decay", type=float, default=1e-5, help="Set the weight decay"
)
parser.add_argument(
    "--lr_warmup", type=int, default=200, help="LR linear warmup iterations"
)
parser.add_argument(
    "--identifier", type=str, default=None, help="model file identifier"
)
parser.add_argument(
    "--steps_per_report",
    type=int,
    default=10,
    help="Number of training steps between loss reporting.",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=50,
    help="Top-k sampling for token output.",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.5,
    help="Temperature for token output after top-k sampling.",
)
parser.add_argument(
    "--steps_per_eval",
    type=int,
    default=1000,
    help="Number of training steps between validations.",
)
parser.add_argument(
    "--eval_test",
    action="store_true",
    help="Evaluate on the test set after training",
)
parser.add_argument(
    "--new_model",
    action="store_true",
    help="Start new model from scratch",
)
parser.add_argument(
    "--inference",
    action="store_true",
    help="Interactive inference mode for testing",
)
args = parser.parse_args()
if not args.gpu:
    mx.set_default_device(mx.cpu)

model_folder = args.save_dir
model_file_extension = ".safetensors"
metadata_file_extension = ".metadata"

finalise_and_exit = threading.Event()
force_exit = threading.Event()
saving_in_progress = threading.Event()
lock = threading.Lock()
ctrl_c_count = 0
ctrl_c_timer = None


def finalize_interrupt():
    global ctrl_c_count
    with lock:
        print("\nCtrl+C detected: will finish current job, then save and exit.")
        ctrl_c_count = 0
        finalise_and_exit.set()


def handle_ctrl_c(signum, frame):
    global ctrl_c_count, ctrl_c_timer
    with lock:
        ctrl_c_count += 1
        if ctrl_c_count == 1:
            if ctrl_c_timer:
                ctrl_c_timer.cancel()
            ctrl_c_timer = threading.Timer(1, finalize_interrupt)
            ctrl_c_timer.start()
        elif ctrl_c_count == 2:
            print(
                "\nDouble Ctrl+C detected: force exit immediately (or after save finishes)."
            )
            if ctrl_c_timer:
                ctrl_c_timer.cancel()
            force_exit.set()
            if not saving_in_progress.is_set():
                sys.exit(1)


signal.signal(signal.SIGINT, handle_ctrl_c)


def save(model, optimizer, data_src, it, exit_after_save=False):
    saving_in_progress.set()

    def _mx2np(mx_tuple: tuple[str, mx.array]) -> Dict[str, np.ndarray]:
        new_dict = {}
        for k, v in mx_tuple:
            new_dict[k] = np.asarray(v)
        return new_dict

    def git_hash_bytes(data_bytes):
        sha1 = hashlib.sha1(data_bytes).hexdigest()
        return sha1

    saved = False
    try:
        flattened_model = tree_flatten(model.trainable_parameters())
        model = _mx2np(flattened_model)
        model_bytes = numpy.save(model)
        model_hash = git_hash_bytes(model_bytes)
        model_extension = ".safetensors"
        metadata_extension = ".metadata"
        new_extension = ".new"
        model_metadata_path = model_folder + "/" + model_hash
        flattened_optimizer_state = tree_flatten(optimizer.state)
        optimizer_state = _mx2np(flattened_optimizer_state)
        training_state = {
            "optimizer": optimizer_state,
            "s": data_src.s,
            "perm": np.asarray(data_src.perm),
            "start": it,
        }

        while not saved:
            try:
                # Critical save section: do NOT abort mid-way
                with open(
                    model_metadata_path + model_extension + new_extension,
                    "wb",
                ) as f:
                    f.write(model_bytes)
                    f.flush()
                    os.fsync(f.fileno())

                with open(
                    model_metadata_path + metadata_extension + new_extension,
                    "wb",
                ) as f:
                    dill.dump(training_state, f)
                    f.flush()
                    os.fsync(f.fileno())  # Ensure all data is written to disk

                safe_open(
                    model_metadata_path + model_extension + new_extension,
                    framework="np",
                )

                with open(
                    model_metadata_path + metadata_extension + new_extension,
                    "rb",
                ) as f:
                    _ = dill.load(f)

            except Exception as e:
                print("Save error:", e)
                # Check force_exit before retrying
                if force_exit.is_set():
                    print(
                        f"\r\nForce exit detected during save error handling. Aborting save."
                    )
                    break  # Abort save immediately

                # Sleep in 1 second chunks to be able to check force_exit regularly
                sleep_seconds = 300  # 5 minutes
                for _ in range(sleep_seconds):
                    if force_exit.is_set():
                        print(
                            f"\r\nForce exit detected during save retry sleep. Aborting save."
                        )
                        break
                    time.sleep(1)

                continue  # retry after sleep

            # If save and load succeeded, break out of retry loop
            saved = True

        if saved:
            # Rename atomically if we got here
            try:
                for name in os.listdir(model_folder):
                    path = os.path.join(model_folder, name)

                    if not os.path.isfile(path):
                        continue

                    if not name.startswith(model_hash) or not name.endswith(
                        new_extension
                    ):  # ensure old model lingering new if exist also get removed, except new model
                        os.remove(path)

            except Exception:
                pass
            finally:
                os.rename(
                    model_metadata_path + model_extension + new_extension,
                    model_metadata_path + model_extension,
                )
                os.rename(
                    model_metadata_path + metadata_extension + new_extension,
                    model_metadata_path + metadata_extension,
                )
                print("Model saved.")
        else:
            print(f"\r\nSave aborted due to forced exit.")
            print(f"\r\nExiting.")
            sys.exit(0)  # Exit immediately
    finally:
        saving_in_progress.clear()
        if exit_after_save:
            print(f"\r\nExiting.")
            sys.exit(0)  # Exit immediately


class ModelFile:
    identifier: str
    ctime: int

    def __init__(self, identifier: str, ctime: int):
        self.identifier = identifier
        self.ctime = ctime


def check_model_file(folder):
    def latest_version(name):
        path = os.path.join(folder, name)
        with safe_open(path, framework="numpy") as f:
            identifier = name.split(".")[0]
            *_, ctime = os.stat(path)
            return ModelFile(identifier, ctime)

    safetensors_files = []
    for name in os.listdir(folder):
        if name.endswith(".safetensors"):
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                model_file = latest_version(name)
                safetensors_files.append(model_file)

    return (
        None if not safetensors_files else max(safetensors_files, key=lambda x: x.ctime)
    )


latest_model_file: Optional[ModelFile] = None
model_file_extension = ".safetensors"
metadata_file_extension = ".metadata"


def load_checkpoint(
    model, optimizer, data_src, model_file_identifier: Optional[str] = None
):
    def _np2mx(np_dict: Dict[str, np.ndarray]) -> list[tuple[str, mx.array]]:
        new_list = []
        for k, v in np_dict.items():
            new_list.append((k, mx.array(v)))
        return new_list

    if model_file_identifier == None:
        print(f"model file not found.")
        return

    target_file_path = model_folder + "/" + model_file_identifier + model_file_extension

    target_metadata_file_path = (
        model_folder + "/" + model_file_identifier + metadata_file_extension
    )

    if args.new_model or not os.path.isfile(target_file_path):
        if args.new_model:
            print(f"Starting with new model from scratch.")
        else:
            print(f"Checkpoint file {target_file_path} not found.")
    else:
        model_flattened_np = numpy.load_file(target_file_path)
        model_flattened = _np2mx(model_flattened_np)
        model_parameters = tree_unflatten(model_flattened)
        model.update(model_parameters)
        with open(target_metadata_file_path, "rb") as f:
            checkpoint = dill.load(f)
            if "optimizer" in checkpoint:
                optimizer_state_flattened = _np2mx(checkpoint["optimizer"])
                optimizer_state = tree_unflatten(optimizer_state_flattened)
                optimizer.state = optimizer_state

            if "s" in checkpoint:
                data_src.s = checkpoint["s"]
                data_src.increment_s()

            if "perm" in checkpoint:
                data_src.perm = mx.array(checkpoint["perm"])

            if "start" in checkpoint:
                return checkpoint["start"]

    return None


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dims: int,
        num_heads: int,
        checkpoint: bool,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.pe = nn.SinusoidalPositionalEncoding(dims)
        self.transformer = nn.TransformerEncoder(
            num_layers, dims, num_heads, norm_first=True, checkpoint=checkpoint
        )
        self.out_proj = nn.Linear(dims, vocab_size)

    def __call__(self, x):
        L = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        x = self.embedding(x)
        x = x + self.pe(mx.arange(L))
        x = self.transformer(x, mask)
        return self.out_proj(x)


def to_samples(context_size, dataset):
    window_size = context_size + 1  # include target
    samples = dataset.size // window_size
    dataset = dataset[: samples * window_size]
    return mx.array(dataset.reshape(samples, -1))


class DataSrc:
    def __init__(self, batch_size, context_size, datasets, s=0):
        self.batch_size = batch_size
        self.context_size = context_size
        self.datasets = datasets
        self.inputs = to_samples(context_size, datasets)
        self.s = s

    def increment_s(self):
        self.s += self.batch_size
        if self.s >= self.inputs.shape[0]:
            self.s = 0

    def iterate_batches(self):
        while True:
            if self.s == 0:
                # Reset permutation:
                self.perm = mx.random.permutation(self.inputs.shape[0])
            ids = self.perm[self.s : self.s + self.batch_size]
            yield self.inputs[ids]
            self.increment_s()


def main(args):
    batch_size = args.batch_size
    context_size = args.context_size
    steps_per_eval = args.steps_per_eval
    steps_per_report = args.steps_per_report

    # Load vocab and dataset:
    vocab, train, valid, test = datasets.load_dataset(
        args.dataset, save_dir=args.save_dir
    )
    to_vocab = {i: v for v, i in vocab.items()}
    data_src = DataSrc(batch_size, context_size, train)

    # Initialize model:
    model = TransformerLM(
        len(vocab), args.num_blocks, args.dim, args.num_heads, args.checkpoint
    )

    model_file_identifier = args.identifier
    if model_file_identifier is None:
        latest_model_file = check_model_file(model_folder)
        if latest_model_file != None:
            model_file_identifier = latest_model_file.identifier

    nparams = sum(
        x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
    )
    print(f"Total parameters: {nparams / 1024**2:.3f} M")

    def to_indices(line):
        line = [w for w in line.strip().split(" ")]
        return np.array(
            [vocab[w] for w in line],
            dtype=np.uint32,
        )

    def loss_fn(model, inputs, reduction="mean"):
        x, y = inputs[..., :-1], inputs[..., 1:]
        logits = model(x)
        return nn.losses.cross_entropy(logits, y, reduction=reduction)

    optimizer = optim.AdamW(
        learning_rate=args.learning_rate, weight_decay=args.weight_decay
    )

    start = load_checkpoint(model, optimizer, data_src, model_file_identifier)
    if start == None:
        start = 0
    train_range = range(start, args.num_iters)
    mx.eval(model.parameters())

    def eval_fn(dataset):
        inputs = to_samples(context_size, dataset)
        loss = 0
        for s in range(0, inputs.shape[0], batch_size):
            losses = loss_fn(model, inputs[s : s + batch_size], reduction="sum")
            loss += losses.item()
        num_samples = inputs.shape[0]
        context_len = inputs.shape[1] - 1
        return loss / (num_samples * context_len)

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(inputs):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, inputs)
        optimizer.update(model, grads)
        return loss

    if not args.inference:
        train_iterator = data_src.iterate_batches()
        losses = []
        tic = time.perf_counter()
        for it, inputs in zip(train_range, train_iterator):
            optimizer.learning_rate = min(1, it / args.lr_warmup) * args.learning_rate
            loss = step(inputs)
            mx.eval(state)
            losses.append(loss.item())
            if (it + 1) % steps_per_report == 0:
                train_loss = sum(losses) / len(losses)
                toc = time.perf_counter()
                print(
                    f"Iter {it + 1}: Train loss {train_loss:.3f}, "
                    f"It/sec {steps_per_report / (toc - tic):.3f}"
                )
                losses = []
                tic = time.perf_counter()
            if (it + 1) % steps_per_eval == 0:
                val_loss = eval_fn(valid)
                toc = time.perf_counter()
                print(
                    f"Iter {it + 1}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val ppl {math.exp(val_loss):.3f}, "
                    f"Val took {(toc - tic):.3f}s, "
                )
                tic = time.perf_counter()

            if (it + 1) % steps_per_eval == 0 or finalise_and_exit.is_set():
                save(model, optimizer, data_src, it + 1, finalise_and_exit.is_set())

                # visualisation
                context = deque(maxlen=args.context_size)
                user_input = "spokeswoman said asbestos was"
                print(">>> " + user_input)
                user_input_indices = to_indices(user_input)
                context.extend(user_input_indices)
                for _ in range(100):  # visualise only first 100 tokens or less
                    x = mx.array(np.array(context))
                    logits = model(x[None, :])
                    token = logits[0, -1, :]
                    top_k = mx.argpartition(-token, kth=args.top_k - 1)[: args.top_k]
                    token = token[top_k]
                    output_logits = nn.log_softmax(token / args.temperature)
                    index = mx.random.categorical(output_logits)
                    detached_index = index.item()
                    top_k_detached_index = top_k[detached_index].item()
                    current_output = to_vocab[top_k_detached_index]
                    if current_output == "<eos>":
                        break
                    context.append(top_k_detached_index)
                    print(" " + to_vocab[top_k_detached_index], end="")

        if args.eval_test:
            test_loss = eval_fn(test)
            test_ppl = math.exp(test_loss)
            print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")
    else:
        context = deque(maxlen=args.context_size)
        while True:
            user_input = input(">>> ")
            if user_input == "quit":
                print("bye bye!")
                break
            user_input_indices = to_indices(user_input)
            context.extend(user_input_indices)
            while True:
                x = mx.array(np.array(context))
                logits = model(x[None, :])
                token = logits[0, -1, :]
                top_k = mx.argpartition(-token, kth=args.top_k - 1)[: args.top_k]
                token = token[top_k]
                output_logits = nn.log_softmax(token / args.temperature)
                index = mx.random.categorical(output_logits)
                detached_index = index.item()
                top_k_detached_index = top_k[detached_index].item()
                current_output = to_vocab[top_k_detached_index]
                if current_output == "<eos>":
                    break
                context.append(top_k_detached_index)
                print(" " + to_vocab[top_k_detached_index], end="")
            print("")


if __name__ == "__main__":
    main(args)
