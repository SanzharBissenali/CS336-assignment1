from collections.abc import Callable
from typing import Any, Callable, Iterable
import torch as torch 
from torch import Tensor
import numpy as np
from einops import rearrange, einsum

class Linear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None): 
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(out_features, in_features, device = device, dtype = dtype)) # Math Notation
        std = np.sqrt(2 / (in_features + out_features))
        torch.nn.init.trunc_normal_(tensor=self.weight, mean=0, std=std, a=-3*std, b=3*std)  # populate the matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")
        return output
    
class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(num_embeddings, embedding_dim, device = device, dtype = dtype))
        torch.nn.init.trunc_normal_(tensor=self.weight, mean=0, std=1, a=-3, b=3)    # populate the matrix

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Here I found a bug, because numpy's uint16 doesn't cast properly;
        # One needs int32 or int64
        return self.weight[x.long()]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, epsilon = 1e-5, device = None, dtype = None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model, device = device, dtype = dtype))
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:     # x is [... d_model]
        rmsNorm = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.epsilon)
        return (x / rmsNorm) * self.weight

class Positionwise_feedforward(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device = None, dtype = None):
        super().__init__()
        self.w1 = Linear(in_features=d_model, out_features=d_ff)
        self.w2 = Linear(in_features=d_ff, out_features=d_model)
        self.w3 = Linear(in_features=d_model, out_features=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        high_w1 = self.w1.forward(x=x)
        silu = high_w1 * torch.sigmoid(high_w1)

        glu = silu * self.w3.forward(x=x)
        out = self.w2.forward(x=glu)
        return out

class RoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k

        # Compute frequencies 
        k = torch.arange(0, d_k // 2, device=device)
        freqs = 1.0 / (theta ** (2 * k / d_k))  # (d_k/2,)

        positions = torch.arange(max_seq_len, device=device).unsqueeze(1)  # (L, 1)

        # Outer product → angles
        angles = positions * freqs  # (L, d_k/2)

        self.register_buffer("cos_tensor", torch.cos(angles))  # (L, d_k/2)
        self.register_buffer("sin_tensor", torch.sin(angles))  # (L, d_k/2)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_tensor[token_positions]   # tokens_positions is usualy [seq_len] -> [0, 1, ... seq_len - 1]
        sin = self.sin_tensor[token_positions]
        
        # NGL, THIS SHIT CONFUSES TO THIS DAY. I WOULD RATHER 
        # IMPLEMENT IT AS A SPARSE MATRIX MULTIPLICATION ... 

        # Rearrange x 
        x = rearrange(x, " ... seq_len (d two) -> ... seq_len d two", two = 2)
        x1, x2 = x[ ..., 0], x[ ..., 1]

        # Apply rotation
        rotated_1 = x1 * cos - x2 * sin
        rotated_2 = x1 * sin + x2 * cos

        # Recombine
        out = torch.stack([rotated_1, rotated_2], dim=-1)
        out = rearrange(out, "... d two -> ... (d two)")

        return out

def softmax(input: torch.Tensor) -> torch.Tensor:
    input = input - torch.amax(input, dim=-1, keepdim = True)
    return torch.exp(input) / torch.sum(torch.exp(input), dim= -1, keepdim=True)

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Note that the dimensions for the inputs are 
    # Q: Float[Tensor, " ... queries d_k"],
    # K: Float[Tensor, " ... keys d_k"],
    # V: Float[Tensor, " ... keys d_v"],
    # mask: Bool[Tensor, " ... queries keys"]
    # In Self-Attention keys = queries, but in general queries != keys, whereas for K and V dim = -2 must match!

    d_k = Q.shape[-1]
    Attn = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / np.sqrt(d_k)  # before masking
    Masked_Attn = softmax(torch.where(mask, Attn, float('-inf')))  # after masking
    return einsum(Masked_Attn, V, "... queries keys, ... keys d_v -> ... queries d_v")

class multihead_self_attention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device = None, dtype = None):
        super().__init__()
        self.q_proj = Linear(in_features=d_model, out_features=d_model, device=device)
        self.k_proj = Linear(in_features=d_model, out_features=d_model, device=device)
        self.v_proj = Linear(in_features=d_model, out_features=d_model, device=device)
        self.output_proj = Linear(in_features=d_model, out_features=d_model, device=device)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.q_proj.forward(x=x)
        K = self.k_proj.forward(x=x)
        V = self.v_proj.forward(x=x)

        Q_rearrange = rearrange(Q, " ... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self.num_heads)
        K_rearrange = rearrange(K, " ... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self.num_heads)
        V_rearrange = rearrange(V, " ... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self.num_heads)
        # mask = torch.tril(torch.ones_like(torch.Tensor((Q.shape[-2], Q.shape[-2]), device=x.device), dtype=torch.bool,), diagonal=0)
        mask = torch.tril(torch.ones(Q.shape[-2], Q.shape[-2], dtype=torch.bool, device=x.device))

        multihead_attn = scaled_dot_product_attention(Q=Q_rearrange, K=K_rearrange, V=V_rearrange, mask= mask)
        multihead_attn = rearrange(multihead_attn, " ... num_heads seq_len d_head -> ... seq_len (num_heads d_head) ")
        return self.output_proj.forward(multihead_attn)
    
class multihead_self_attention_with_RoPE(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, device = None, dtype = None):
        super().__init__()
        self.q_proj = Linear(in_features=d_model, out_features=d_model, device=device)
        self.k_proj = Linear(in_features=d_model, out_features=d_model, device=device)
        self.v_proj = Linear(in_features=d_model, out_features=d_model, device=device)
        self.output_proj = Linear(in_features=d_model, out_features=d_model, device=device)
        self.num_heads = num_heads

        self.rope = RoPE(theta=theta, max_seq_len=max_seq_len, d_k= int(d_model/num_heads ), device=device)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        Q = self.q_proj.forward(x=x)
        K = self.k_proj.forward(x=x)
        V = self.v_proj.forward(x=x)

        Q_rearrange = rearrange(Q, " ... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self.num_heads)
        K_rearrange = rearrange(K, " ... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self.num_heads)
        V_rearrange = rearrange(V, " ... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads = self.num_heads)
        # mask = torch.tril(torch.ones_like(torch.Tensor(Q.shape[-2], Q.shape[-2]), dtype=torch.bool), diagonal=0)

        mask = torch.tril(torch.ones(Q.shape[-2], Q.shape[-2], dtype=torch.bool, device=x.device))

        # Now apply the RoPE inside each head; the bug was that you applied it to big Q and K.
        Q_rearrange = self.rope.forward(x=Q_rearrange, token_positions=token_positions)
        K_rearrange = self.rope.forward(x=K_rearrange, token_positions=token_positions)

        multihead_attn = scaled_dot_product_attention(Q=Q_rearrange, K=K_rearrange, V=V_rearrange, mask= mask)
        multihead_attn = rearrange(multihead_attn, " ... num_heads seq_len d_head -> ... seq_len (num_heads d_head) ")
        return self.output_proj.forward(multihead_attn)
    
class transformer_block(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        self.attn = multihead_self_attention_with_RoPE(d_model=d_model, num_heads=num_heads, max_seq_len=max_seq_len, 
                                                       theta=theta)
        self.ffn = Positionwise_feedforward(d_model=d_model, d_ff=d_ff)
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre_attn = self.ln1.forward(x=x)
        post_attn = x + self.attn.forward(x=pre_attn, token_positions= torch.arange(x.shape[-2], device=x.device))

        pre_ffn = self.ln2.forward(x=post_attn)
        out = post_attn + self.ffn.forward(x=pre_ffn)

        return out

class transformer_lm(torch.nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_layers: int, num_heads: int, 
                 d_ff: int, max_seq_len: int, rope_theta: float, ):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.layers = torch.nn.ModuleList(transformer_block(d_model=d_model, num_heads=num_heads, d_ff=d_ff,
                                        max_seq_len=max_seq_len, theta=rope_theta) for _ in range(num_layers))
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Firstly, we do embedding
        z = self.token_embeddings.forward(x=x)

        # Pass through num_layers of transformer blocks
        for i in range(self.num_layers):
            z = self.layers[i].forward(x=z)

        # Apply the final Normalization and Output embedding 
        out = self.lm_head.forward(x=self.ln_final.forward(x=z))
        return out

def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """
    # Firstly, to handle numerical issues, from every row subtract the largest value 
    inputs = inputs - torch.amax(inputs, dim=-1, keepdim=True)

    # Secondly, let's extract the raw logits score for the correct tokens 
    targets = targets.view(-1, 1)
    raw_logits_correct = inputs.gather(1, targets).squeeze(dim=-1)

    # Thirdly, let's compute the log term
    log_term = torch.log(torch.sum(torch.exp(inputs), dim=-1))  # [batch_size]
    out = torch.mean(-(raw_logits_correct) + log_term)  # [number]

    return out

def learning_rate_schedule(it: int, max_learning_rate: float, min_learning_rate: float, 
                           warmup_iters: int, cosine_cycle_iters: int,):
    if it <= warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    elif it >= cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + np.cos( np.pi * (it - warmup_iters)/(cosine_cycle_iters - warmup_iters)))

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    parameters = list(parameters)
    device = parameters[0].device
    Norm = torch.tensor([0.0], device=device)
    for parameter in parameters:
        if parameter.grad is None:
            continue
        Norm += torch.sum((parameter.grad)**2)
    Norm = torch.sqrt(Norm)

    if Norm > max_l2_norm:
        factor = max_l2_norm / (Norm + 1e-6)
        for parameter in parameters:
            if parameter.grad is None:
                continue
            parameter.grad *= factor
    return None

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, Tensor]], 
                 lr = 1e-3, weight_decay = 0.01, betas = (0.9, 0.95), eps = 1e-8) -> None:
        defaults = {"lr" : lr, 
                         "weight_decay" : weight_decay, 
                         "beta1" : betas[0], 
                         "beta2" : betas[1],
                         "eps" : eps}
        super().__init__(params, defaults)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] 
            beta1 = group["beta1"] 
            beta2 = group["beta2"] 
            weight_d = group['weight_decay'] 
            eps = group['eps']

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]                           # Get state associated with p.
                t = state.get("t", 1)                           # Get iteration number from the state, or 0.
                grad = p.grad.data                              # Get the gradient of loss with respect to p.

                adjusted_lr = lr * np.sqrt(1 - beta2**t) / (1 - beta1**t)
                p.data -= lr * weight_d * p.data                  # Update weight decay.

                m = state.get("m", 0)                           # Get iteration number from the state, or 0.
                v = state.get("v", 0)                           # Get iteration number from the state, or 0.

                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad**2

                p.data -= adjusted_lr * m / (torch.sqrt(v) + eps)     # Update AdamW

                state["t"] = t + 1                              # Increment iteration number.
                state["m"] = m                                  # Update m and v
                state["v"] = v  
            
        return loss


    

