import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
import tiktoken
from dataclasses import dataclass
import time, os


@dataclass
class GPTConfig:
    block_size : int = 1024
    vocab_size : int = 50257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention used instead of traditional to increase speed
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

        self.c_proj.SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # adding x to the output of attention and mlp for residual connection
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme as mentioned in GPT-2
        self.transformer.wte.weight = self.lm_head.weight

        # apply function recursively applies the function to each submodule
        self.apply(self._init_weights) 

    def _init_weights(self, submodule):
        std = 0.02
        if isinstance(submodule, nn.Linear):
            if hasattr(submodule, "SCALE_INIT"):
                # multiplied by 2 because the residual layer is connected 2 times in each block
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(submodule.weight, mean=0, std=std)
            if submodule.bias is not None:
                torch.nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, nn.Embedding):
            torch.nn.init.normal_(submodule.weight, mean=0, std=std)
    
    def forward(self, idx, target=None):
        B, T = idx.size()
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = pos_emb + tok_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return logits, loss
    
    def configure_optimizer(self, weight_decay=0.1, betas=(0.9, 0.95), lr=6e-4, eps=1e-8):
        param_dict = { name:param for (name, param) in self.named_parameters() }
        param_dict = { name:param for (name, param) in param_dict.items() if param.requires_grad }

        # not applying weight decay to biases, layer norm, other 1-D params etc.
        decay_params = [param for name, param in param_dict.items() if param.dim() >= 2]
        non_decay_params = [param for name, param in param_dict.items() if param.dim() < 2]

        num_decay_params = sum([param.numel() for param in decay_params])
        num_non_decay_params = sum([param.numel() for param in non_decay_params])
        print(f"Number of decay params tensor: {len(decay_params)} with {num_decay_params} parameters")
        print(f"Number of non-decay params tensor: {len(non_decay_params)} with {num_non_decay_params} parameters")

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": non_decay_params, "weight_decay": 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, betas=betas, lr=lr, eps=eps) # replicating GPT-2 optimizer
        return optimizer
 

class DataLoaderLite:
    
    def __init__(self, B, T, process_rank, num_process, train=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_process = num_process
        split = "train" if train else "val"
        enc = tiktoken.get_encoding("gpt2")
        with open("input.txt", "r") as f: 
            data = f.read()
        data = data[:int(len(data) * 0.8)] if train else data[int(len(data) * 0.8):]
        self.tokens = enc.encode(data)
        self.current_pos = B * T * process_rank
        print(f"Number of tokens: {len(self.tokens)}")
        print(f"1 epoch for {split} has {len(self.tokens) // (B * T * num_process)} batches")
    
    def next_batch(self):
        buf = torch.tensor(self.tokens[self.current_pos: self.current_pos + self.B * self.T + 1])
        x = buf[:-1].view(self.B, self.T)
        y = buf[1:].view(self.B, self.T)
        self.current_pos += self.B * self.T * self.num_process
        if self.current_pos + self.B * self.T * self.num_process + 1 > len(self.tokens):
            self.current_pos = self.B * self.T * self.process_rank
        return x, y

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")

ddp = int(os.getenv("RANK", -1)) != -1
if  ddp and torch.cuda.is_available():
    # initializing CUDA software library
    init_process_group(backend="nccl")
    # global rank of a GPU in case of multiple nodes
    ddp_rank = int(os.environ["RANK"])
    # local rank of a GPU on each node
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    # total number of GPUs on all the nodes
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_prcess = ddp_rank == 0 # used for logging, checkpointing etc
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_prcess = True

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Device: {device}")

torch.manual_seed(42)
if device == "mps":
    torch.mps.manual_seed(42)
elif device == "cuda": 
    torch.cuda.manual_seed(42)

# each GPU creating its own model
model = GPT(GPTConfig(vocab_size=50304)) # making vocab size being divisible by powers of 2 to increase speed
model.to(device)

if device == "cuda":
    backend = "inductor"
elif device == "mps":
    backend = "aot_eager"
if device != "cpu":
    model = torch.compile(model, backend=backend)

if ddp:
    # registering the model for parallelism to do all reduce when necessary (like gradient synchronization)
    model = DDP(module=model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # used when we are directly dealing with the model rather than a DDP registered

total_batch_size = 524288
B, T = 4, 1024
train_dataloader = DataLoaderLite(B, T, process_rank=ddp_rank, num_process=ddp_world_size)
val_dataloader = DataLoaderLite(B, T, process_rank=ddp_rank, num_process=ddp_world_size, train=False)
max_steps = 100
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

# tf32 matrix multiplication to increase speed
torch.set_float32_matmul_precision("high")
optimizer = raw_model.configure_optimizer(weight_decay=0.1, betas=(0.9, 0.95), lr=6e-4, eps=1e-8)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

for step in range(max_steps):
    t0 = time.time()
    if step % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_loss_accum = 0
            val_accum_steps = 4
            for val_step in range(val_accum_steps):
                x, y = val_dataloader.next_batch()
                x, y = x.to(device), y.to(device) 
                with torch.autocast(device_type=device, dtype=torch.float16):
                    logits, loss = model(x, y)
                loss = loss / val_accum_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_prcess:
            print(f"Val loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"Step: {step} with Validation loss: {val_loss_accum.item():.4f}\n")
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                torch.save(checkpoint, checkpoint_path)
    
    if not model.training:
        model.train()
    optimizer.zero_grad()
    loss_accum = 0
    with model.no_sync():
        # no gradient sync between the GPUs before the last gradient step (optional since how frequent grad sync happens is upto us)
        for grad_step in range(grad_accum_steps - 1):
            x, y = train_dataloader.next_batch()
            x, y = x.to(device), y.to(device) 
            # some parameters remain at float32 while others like loss become float16. Used to increase speed
            with torch.autocast(device_type=device, dtype=torch.float16):
                logits, loss = model(x, y)
            loss = loss / grad_accum_steps
            loss_accum += loss.detach() # detach to let pytorch know this does not need to be kept in gradient compute graph
            loss.backward()
    x, y = train_dataloader.next_batch()
    x, y = x.to(device), y.to(device) 
    # some parameters remain at float32 while others like loss become float16. Used to increase speed
    with torch.autocast(device_type=device, dtype=torch.float16):
        logits, loss = model(x, y)
    loss = loss / grad_accum_steps
    loss_accum += loss.detach() # detach to let pytorch know this does not need to be kept in gradient compute graph
    loss.backward()
    if ddp:
        # all reduce by taking average on loss accumulated by each GPU
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # clipping gradient norm to not exceed 1 due to sudden increase in loss
    optimizer.step()
    scheduler.step()
    # CPU does not record time till everything running on GPU is complete 
    torch.mps.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000
    tok_per_sec = int(train_dataloader.B * train_dataloader.T * grad_accum_steps * ddp_world_size / (t1 - t0))
    if master_prcess:
        print(f"Step: {step} | loss: {loss_accum.item():.4f} | norm: {norm:.4f} | Time: {dt:.2f}ms | Tokens/s: {tok_per_sec}")

if ddp:
    destroy_process_group()

num_return_sequences = 5
max_length = 30

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Donald Trump is a")
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
x = tokens.to(device)

# generating random numbers for each GPU to get different multinomial output
rng = torch.Generator(device=device)
rng.manual_seed(42 + ddp_rank)

model.eval()

while x.size(1) <= max_length:
    with torch.no_grad():
        logits, _ = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        idx_next = torch.multinomial(topk_probs, 1, generator=rng)
        # mapping index from multinomial (0 to 49) to indexes of top k containing actual token ids
        xcol = torch.gather(topk_indices, -1, idx_next)
        x = torch.cat((x, xcol), dim=-1)

for i in range(num_return_sequences):
    tokens = x[i].tolist()
    decoded = enc.decode(tokens)
    print(f"Rank {ddp_rank} > {decoded}")