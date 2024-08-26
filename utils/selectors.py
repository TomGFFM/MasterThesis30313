import torch.nn.functional as F
from typing import Dict
from torch import nn, optim
from torch.optim.lr_scheduler import LRScheduler


class OptimizerSelector(object):
    """
    A class for selecting the appropriate PyTorch optimizer based on the given hyperparameters.

    Methods
    -------
    __call__(agent_hyper_params: Dict, network: nn.Module) -> optim.Optimizer:
        Returns a PyTorch optimizer based on the specified optimizer name in the hyperparameters.
    """

    def __call__(self, agent_hyper_params: Dict, network: nn.Module) -> optim.Optimizer:
        """
        Selects and returns a PyTorch optimizer based on the hyperparameters provided.

        Args:
            agent_hyper_params (Dict): A dictionary containing the optimizer name and relevant hyperparameters.
            network (nn.Module): The neural network whose parameters will be optimized.

        Returns:
            optim.Optimizer: The selected PyTorch optimizer.

        Raises:
            ValueError: If the optimizer name is not supported.
        """
        if agent_hyper_params['optimizer_name'] == "Adam":
            optimizer = optim.Adam(network.parameters(), lr=agent_hyper_params['learning_rate'])
        elif agent_hyper_params['optimizer_name'] == "NAdam":
            optimizer = optim.NAdam(network.parameters(), lr=agent_hyper_params['learning_rate'])
        elif agent_hyper_params['optimizer_name'] == "SGD":
            optimizer = optim.SGD(network.parameters(), lr=agent_hyper_params['learning_rate'], momentum=0.9,
                                  nesterov=True)
        elif agent_hyper_params['optimizer_name'] == "Adagrad":
            optimizer = optim.Adagrad(network.parameters(), lr=agent_hyper_params['learning_rate'])
        elif agent_hyper_params['optimizer_name'] == "Adadelta":
            optimizer = optim.Adadelta(network.parameters(), lr=1.0)
        elif agent_hyper_params['optimizer_name'] == "RAdam":
            optimizer = optim.RAdam(network.parameters(), lr=agent_hyper_params['learning_rate'])
        elif agent_hyper_params['optimizer_name'] == "RMSprop":
            optimizer = optim.RMSprop(network.parameters(), lr=agent_hyper_params['learning_rate'])
        else:
            raise ValueError(f"Optimizer {agent_hyper_params['optimizer_name']} not supported")

        return optimizer


class LRSchedulerSelector(object):
    """
    A class for selecting the appropriate learning rate scheduler based on the given hyperparameters.

    Methods
    -------
    __call__(agent_hyper_params: Dict, network: nn.Module, optimizer: optim.Optimizer) -> LRScheduler:
        Returns a PyTorch learning rate scheduler based on the specified scheduler name in the hyperparameters.
    """

    def __call__(self, agent_hyper_params: Dict, optimizer: optim.Optimizer) -> LRScheduler:
        """
        Selects and returns a PyTorch learning rate scheduler based on the hyperparameters provided.

        Args:
            agent_hyper_params (Dict): A dictionary containing the scheduler name and relevant hyperparameters.
            optimizer (optim.Optimizer): The optimizer associated with the network.

        Returns:
            LRScheduler: The selected PyTorch learning rate scheduler or None if no scheduler is selected.

        Raises:
            ValueError: If the scheduler name is not supported.
        """
        if agent_hyper_params['lr_scheduler_name'] == "cosine":
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=agent_hyper_params['n_episodes'],
                                                                eta_min=0.000001)
        elif agent_hyper_params['lr_scheduler_name'] == "step":
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                     step_size=agent_hyper_params['learning_rate_step_size'],
                                                     gamma=agent_hyper_params['learning_rate_gamma'])
        elif agent_hyper_params['lr_scheduler_name'] == "reduce_on_plateau":
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        elif agent_hyper_params['lr_scheduler_name'] == "none":
            lr_scheduler = None
        else:
            raise ValueError(f"LR scheduler {agent_hyper_params['lr_scheduler_name']} not supported")

        return lr_scheduler


class LossFunctionSelector(object):
    """
    A class for selecting the appropriate loss function based on the given hyperparameters.

    Methods
    -------
    __call__(agent_hyper_params: Dict) -> callable:
        Returns a PyTorch loss function based on the specified loss function name in the hyperparameters.
    """

    def __call__(self, agent_hyper_params: Dict) -> callable:
        """
        Selects and returns a PyTorch loss function based on the hyperparameters provided.

        Args:
            agent_hyper_params (Dict): A dictionary containing the loss function name.

        Returns:
            callable: The selected PyTorch loss function.

        Raises:
            ValueError: If the loss function name is not supported.
        """
        if agent_hyper_params['loss_name'] == "huber":
            loss_function = F.smooth_l1_loss
        elif agent_hyper_params['loss_name'] == "mse":
            loss_function = F.mse_loss
        elif agent_hyper_params['loss_name'] == "l1":
            loss_function = F.l1_loss
        else:
            raise ValueError(f"Loss function {agent_hyper_params['loss_name']} not supported")

        return loss_function
