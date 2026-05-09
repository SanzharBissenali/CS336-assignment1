import torch as torch
import numpy as np 
import numpy.typing as npt

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    start_indexes = np.random.randint(0, len(dataset) - context_length, 
                                      size = batch_size).reshape(-1, 1)
    offsets = np.arange(context_length)

    input_ind = start_indexes + offsets
    label_ind = input_ind + 1

    input = torch.tensor(dataset[input_ind], device=device)
    label = torch.tensor(dataset[label_ind], device=device)
    
    return (input, label)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str):
    dictionary_to_save = {
        "model" : model.state_dict(), 
        "optimizer" : optimizer.state_dict(),
        "iteration" : iteration
    }
    torch.save(dictionary_to_save, out)
    return None

def load_checkpoint(src: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    dictionary_loaded = torch.load(src)
    model.load_state_dict(dictionary_loaded['model'])
    optimizer.load_state_dict(dictionary_loaded['optimizer'])
    return dictionary_loaded['iteration']

def get_specific_batch(dataset: npt.NDArray, batch_size: int, context_length: int, start_index: int, device: str)-> tuple[torch.Tensor, torch.Tensor]:
    stop_index = start_index + batch_size * context_length
    start_indexes = np.arange(start_index, stop_index, context_length).reshape(-1, 1)

    offsets = np.arange(context_length)

    input_ind = start_indexes + offsets
    label_ind = input_ind + 1

    input = torch.tensor(dataset[input_ind], device=device)
    label = torch.tensor(dataset[label_ind], device=device)

    return (input, label)






    







