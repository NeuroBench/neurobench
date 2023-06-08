import os

from typing import Union, Callable, Dict, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

import learn2learn as l2l
import tqdm

from torch_mate.utils import calc_accuracy, nested_tuple_to_device, set_seeds
from torch_mate.contexts import evaluating

from neurobench.utils import Dict2Class

def gradient_based_fast_adapt(learner: nn.Module, loss: nn.Module, sample: Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]], adaptation_steps: int) -> Tuple[Tensor, Tensor]:
    ((adaptation_data, evaluation_data), (adaptation_labels, evaluation_labels)) = sample

    # Adapt the model
    for _ in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    # Evaluate the adapted model
    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)

    valid_accuracy = calc_accuracy(predictions, evaluation_labels)

    return valid_error, valid_accuracy

def train_using_maml(
    model: nn.Module,
    train_data_loader: torch.utils.data.DataLoader,
    test_data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    model_save_directory: str,
    cfg: Dict2Class,
    log: Union[Callable[[Dict], None], None] = None, 
    test_every=20,
    save_every=1000
):
    set_seeds(cfg.seed, device)

    model.to(device)

    os.makedirs(model_save_directory)

    if log == None:
        log = lambda x: None
 
    maml = l2l.algorithms.MAML(model, lr=cfg.meta_learning.fast_lr, first_order=cfg.meta_learning.first_order)
    maml.to(device)

    opt = getattr(optim, cfg.optimizer.name)(model.parameters(), **cfg.optimizer.cfg.toDict())

    iterable_test_data_loader = iter(test_data_loader)

    loss = getattr(nn, cfg.criterion.name)()

    for iteration, meta_batch in tqdm(zip(range(cfg.meta_learning.num_iterations), train_data_loader)):
        opt.zero_grad(set_to_none=True)

        ((X_train, X_test), (y_train, y_test)) = nested_tuple_to_device(meta_batch, device)

        meta_train_error = 0.0
        meta_train_accuracy = 0.0

        for task in range(cfg.meta_learning.meta_batch_size):
            # Compute meta-training loss
            learner = maml.clone()

            evaluation_error, evaluation_accuracy = gradient_based_fast_adapt(learner,
                                                                              loss,
                                                                              ((X_train[task], X_test[task]), (y_train[task], y_test[task])),
                                                                              cfg.meta_learning.adaptation_steps)
            evaluation_error.backward()

            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

        # Average the accumulated gradients and optimize
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1.0 / cfg.meta_learning.meta_batch_size)
        opt.step()

        log({"iteration": iteration, "meta_train/loss": meta_train_error / cfg.meta_learning.meta_batch_size, "meta_train/accuracy": meta_train_accuracy / cfg.meta_learning.meta_batch_size})

        if iteration % test_every == 0 or iteration == cfg.meta_learning.num_iterations -1:
            meta_batch = next(iterable_test_data_loader)
            
            ((X_train, X_test), (y_train, y_test)) = nested_tuple_to_device(meta_batch, device)

            meta_test_error = 0.0
            meta_test_accuracy = 0.0

            # Using evaluating(model) or evaluating(maml) here does not make a difference
            with evaluating(model):
                for task in range(cfg.meta_learning.meta_batch_size):
                    # Compute meta-testing loss
                    learner = maml.clone()

                    evaluation_error, evaluation_accuracy = gradient_based_fast_adapt(learner,
                                                                    loss,
                                                                    ((X_train[task], X_test[task]), (y_train[task], y_test[task])),
                                                                    cfg.meta_learning.test_adaptation_steps)

                    meta_test_error += evaluation_error.item()
                    meta_test_accuracy += evaluation_accuracy.item()

            log({"iteration": iteration, "meta_test/loss": meta_test_error / cfg.meta_learning.meta_batch_size, "meta_test/accuracy":  meta_test_accuracy / cfg.meta_learning.meta_batch_size})

        if iteration % save_every == 0 or iteration == cfg.meta_learning.num_iterations -1:
            torch.save(model.state_dict(), f"{model_save_directory}/iteration_{iteration}.pth")
