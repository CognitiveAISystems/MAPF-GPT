import multiprocessing
from copy import deepcopy

import pyarrow as pa

# Ensure the 'spawn' start method is used
multiprocessing.set_start_method("spawn", force=True)
import glob
import math
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from itertools import cycle

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt.model import GPT, GPTConfig
from tokenizer.parameters import InputParameters
from tokenizer.tokenizer import Tokenizer

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = "out"
eval_interval = 500
log_interval = 1
eval_iters = 40
always_save_checkpoint = True  # if True, always save a checkpoint after each eval
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True  # disabled by default
wandb_project = "mapf-gpt"
# wandb_run_name = 'mini-mapf-gpt' # 'run' + str(time.time())
# data
dataset = "mazes-82M"
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
batch_size = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 512
# model
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = "nccl"  # 'nccl', 'gloo', etc.
# system
device = (
    "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
)
dtype = (
    "bfloat16"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else "float16"
)  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
train_data_file = "dataset/mazes-castar.hdf5"
valid_data_file = "dataset/mazes-castar-val.hdf5"

config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("gpt/configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
current_train_index = 0
current_valid_index = 0


def get_dataset_size(folder_path):
    file_paths = glob.glob(f"{folder_path}/*.arrow")
    total_elements = 0

    for file_path in file_paths:
        with pa.memory_map(file_path) as source:
            table = pa.ipc.open_file(source).read_all()
        num_elements = len(table)
        total_elements += num_elements

    return total_elements


def calculate_epochs(
    max_iters, dataset_size, batch_size, gradient_accumulation_steps=1
):
    # Effective batch size considering gradient accumulation
    effective_batch_size = batch_size * gradient_accumulation_steps

    # Number of steps (iterations) to go through the entire dataset once (i.e., one epoch)
    steps_per_epoch = dataset_size // effective_batch_size

    # Total number of epochs
    num_epochs = max_iters / steps_per_epoch

    return num_epochs


def human_readable_size(size):
    for unit in ["pairs", "K pairs", "M pairs", "B pairs"]:
        if size < 1000:
            return f"{size:.2f} {unit}"
        size /= 1000
    return f"{size:.2f} B pairs"


print(f"Train set size: {human_readable_size(get_dataset_size(train_data_file))}")
print(f"Validation set size: {human_readable_size(get_dataset_size(valid_data_file))}")

# Calculate the number of epochs
num_epochs = calculate_epochs(
    max_iters,
    get_dataset_size(train_data_file),
    batch_size,
    gradient_accumulation_steps,
)

print(f"Number of training epochs: {num_epochs:.2f}")

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)


def mask(batch):
    result = []
    for input in batch:
        result.append(tokenizer.encoder.mask(input))
    return result


def batch_generator(path):
    file_paths = sorted(glob.glob(os.path.join(path, "*.arrow")))
    total_files = len(file_paths)
    if "train" in path:
        files_per_worker = total_files // ddp_world_size
        start_index = ddp_local_rank * files_per_worker
        end_index = start_index + files_per_worker
        file_paths = file_paths[start_index:end_index]

    def read_arrow_files(file_paths):
        for file_path in cycle(file_paths):
            with pa.memory_map(file_path) as source:
                table = pa.ipc.open_file(source).read_all()
            input_tensors = table["input_tensors"].to_numpy()
            input_tensors = np.stack(input_tensors).astype(np.int64)
            gt_actions = table["gt_actions"].to_numpy()
            gt_actions = np.array(gt_actions, dtype=np.int64)

            indices = np.arange(len(input_tensors))
            np.random.shuffle(indices)

            shuffled_input_tensors = input_tensors[indices]
            shuffled_gt_actions = gt_actions[indices]
            shuffled_input_tensors = torch.tensor(shuffled_input_tensors, dtype=torch.long).to(device, non_blocking=True)
            shuffled_gt_actions = torch.full_like(shuffled_input_tensors, -1, dtype=torch.long)
            shuffled_gt_actions[:, -1] = torch.tensor(gt_actions, dtype=torch.long)
            shuffled_gt_actions = shuffled_gt_actions.to(device, non_blocking=True)
            yield shuffled_input_tensors, shuffled_gt_actions

    for input_tensors, gt_actions in read_arrow_files(file_paths):
        num_elements = len(input_tensors)
        start_index = 0

        while start_index + batch_size <= num_elements:
            end_index = start_index + batch_size
            data = input_tensors[start_index:end_index]
            if (
                cfg.mask_cost2go
                or cfg.mask_goal
                or cfg.mask_greed_action
                or cfg.mask_actions_history
            ):
                data = mask(deepcopy(data))
            yield {
                "input_tensors": data,
                "gt_actions": gt_actions[start_index:end_index],
            }
            start_index = end_index


cfg = InputParameters()
tokenizer = Tokenizer(cfg)
train_data = batch_generator(train_data_file)
val_data = batch_generator(valid_data_file)


def get_batch(data):
    batch = next(data)
    return batch["input_tensors"], batch["gt_actions"]


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
# meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = len(tokenizer.encoder.vocab)
# if os.path.exists(meta_path):
#     with open(meta_path, 'rb') as f:
#         meta = pickle.load(f)
#     meta_vocab_size = meta['vocab_size']
#     print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=None,
    dropout=dropout,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    model_args["vocab_size"] = meta_vocab_size
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]

print("number of parameters: %.2fM" % (model.get_num_params() / 1e6,))

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type
)
if init_from == "resume":
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory
# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            if split == "train":
                X, Y = get_batch(train_data)
            else:
                X, Y = get_batch(val_data)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, config=config)

# training loop
X, Y = get_batch(train_data)  # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        model.eval()
        losses = estimate_loss()
        model.train()
        print(
            f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        if wandb_log:
            wandb.log(
                {
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                }
            )
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (
                micro_step == gradient_accumulation_steps - 1
            )
        with ctx:
            logits, loss = model(X, Y)
            loss = (
                loss / gradient_accumulation_steps
            )  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch(train_data)
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        # print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
        print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%, {iter_num} / {max_iters}"
        )

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
