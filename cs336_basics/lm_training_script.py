import os
import torch as torch
import numpy as np
import argparse
import wandb
from einops import rearrange
from datetime import datetime

from language_model import transformer_lm, AdamW, cross_entropy_loss, learning_rate_schedule, gradient_clipping
from data_preprocessing import get_batch, save_checkpoint

# 1. Initialize the parser
parser = argparse.ArgumentParser(description="Machine Learning Training Script")

# 2. Add arguments
parser.add_argument("--d_model", type=int, default=16, help="Residual Stream Dimension")
parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer Layers")
parser.add_argument("--d_ff", type=int, default=16*4, help="FFN Dimension")
parser.add_argument("--n_heads", type=int, default=4, help="Number of Attention Heads")
parser.add_argument("--rope_theta", type=float, default=np.pi/360, help="RoPE Theta Value")
parser.add_argument("--max_seq_len", type=int, default=1024, help="Max Sequence Length")

parser.add_argument("--max_lr", type=float, default=1e-3, help="Max Learning Rate")
parser.add_argument("--min_lr", type=float, default=1e-5, help="Min Learning Rate")
parser.add_argument("--weight_decay", type=float, default=1e-5, help="Min Learning Rate")
parser.add_argument("--warmup", type=int, default=350, help="Warmup Iters")
parser.add_argument("--cosine", type=int, default=6650, help="Cosine Iters")
parser.add_argument("--num_steps", type=int, default=7000, help="Total Num of Steps")
parser.add_argument("--batch_size", type=int, default=64, help="Batch Size")

parser.add_argument("--output_dir", type=str, default = "runs", help="Output Directory where to Save")

# 3. Parse arguments
args = parser.parse_args()

run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="models-california-institute-of-technology-caltech",
    # Set the wandb project where this run will be logged.
    project="CS-336-1",
    # Track hyperparameters and run metadata.
    config={
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "d_ff": args.d_ff,
        "num_heads": args.n_heads,
        "rope_theta": args.rope_theta,
        "max learning_rate": args.max_lr,
        "architecture": "Transformer",
        "dataset": "TinyStories",
        "steps": args.num_steps,
        "warmup": args.warmup,
        "batch_size": args.batch_size,
    },
)

# Now, let's implement the Training Loop !!!
model = transformer_lm(vocab_size=10000, d_model= args.d_model, num_layers=args.num_layers, num_heads= args.n_heads,
                       d_ff= args.d_ff, max_seq_len= args.max_seq_len, rope_theta= args.rope_theta)

optimizer = AdamW(params=model.parameters(), weight_decay=args.weight_decay)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print('The device is', device)

train_ids = np.load('output/np_training_set.npy', mmap_mode='r')
val_ids = np.load('output/np_validation_set.npy', mmap_mode='r')

num_steps = args.num_steps
batch_size = args.batch_size
max_l2_norm = 1
model.train()
model.to(device=device)

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
hyperparam_str = f"d{args.d_model}_l{args.num_layers}_h{args.n_heads}_lr{args.max_lr}"
run_name = hyperparam_str + "_" + timestamp_str
dir_path = os.path.join(args.output_dir, run_name)
os.makedirs(dir_path, exist_ok=True)

for i in range(num_steps):
    optimizer.zero_grad()

    lr = learning_rate_schedule(it=i, max_learning_rate= args.max_lr, min_learning_rate=args.min_lr,
                                warmup_iters=args.warmup, cosine_cycle_iters= args.cosine)
    optimizer.param_groups[0]['lr'] = lr

    input, label = get_batch(dataset=train_ids, batch_size=batch_size, context_length=args.max_seq_len, device=device)

    output = model.forward(input)
    # Another bug that I found - have to reshape the 3D output of the model to feed into
    # the cross_entropy_loss function

    output = rearrange(output, "batch_size seq_len vocab_size -> (batch_size seq_len) vocab_size").to(device=device)
    label = rearrange(label, "batch_size seq_len -> (batch_size seq_len)").to(device=device)

    loss = cross_entropy_loss(inputs=output, targets=label.long())

    loss.backward()
    gradient_clipping(parameters=model.parameters(), max_l2_norm=max_l2_norm)
    optimizer.step()

    if i % 1000 == 0:
        file_path = os.path.join(dir_path, f"checkpoint{i}.pt")
        save_checkpoint(model=model, optimizer=optimizer, iteration= i, out= file_path)
    if i % 100 == 0:
        model.eval()
        with torch.no_grad():
            val_input, val_label = get_batch(dataset=val_ids, batch_size=batch_size, context_length=args.max_seq_len, device=device)
            output = model.forward(val_input)

            output = rearrange(output, "batch_size seq_len vocab_size -> (batch_size seq_len) vocab_size").to(
                device=device)
            val_label = rearrange(val_label, "batch_size seq_len -> (batch_size seq_len)").to(device=device)

            val_loss = cross_entropy_loss(inputs=output, targets=val_label.long())
            print(f"At training step {i}, Training Loss is {loss.item()}, and Validation loss is {val_loss.item()}. lr is {lr}.")
            run.log({"train_loss": loss.item(), "val_loss": val_loss.item(), "lr": lr}, step=i)
        model.train()

# Finish the run and upload any remaining data.
run.finish()
