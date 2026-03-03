import torch
from torch import nn


class Accumulator:
    """Accumulator class for accumulating multiple scalar values."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, key):
        return self.data[key]


def accuracy(y_hat, y):
    """Calculate the number of correct predictions.

    Args:
        y_hat: Model predictions with shape (batch_size, num_classes)
        y: Ground truth labels with shape (batch_size,)

    Returns:
        float: Number of correct predictions
    """
    y_hat = y_hat.argmax(axis=1)
    cmp = y_hat == y
    return cmp.float().sum().item()


def evaluator(net, test_iter):
    """Evaluate model accuracy on the test set.

    Args:
        net: Neural network model
        test_iter: Test data iterator

    Returns:
        float: Accuracy on the test set
    """
    if isinstance(net, nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in test_iter:
            y_hat = net(X)
            acc = accuracy(y_hat, y)
            metric.add(acc, y.numel())
    return metric[0] / metric[1]


def train_epoch(model, data_iter, loss_function, optimizer):
    """Train for one epoch.

    Args:
        model: Neural network model
        data_iter: Training data iterator
        loss_function: Loss function
        optimizer: Optimizer

    Returns:
        float: Training accuracy (as a percentage)
    """
    if isinstance(model, nn.Module):
        model.train()
    metric = Accumulator(2)
    for X, y in data_iter:
        y_hat = model(X)
        loss = loss_function(y_hat, y)
        if isinstance(optimizer, torch.optim.Optimizer):
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
        else:
            loss.sum().backward()
            optimizer(X.shape[0])
        metric.add(accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1]


def train(model, data_iter, loss_function, optimizer, epochs, test_iter):
    """Train the model for multiple epochs.

    Args:
        model: Neural network model
        data_iter: Training data iterator
        loss_function: Loss function
        optimizer: Optimizer
        epochs: Number of training epochs
        test_iter: Test data iterator

    Returns:
        tuple: (train_acc_list, test_acc_list) - lists of training and test accuracies
    """
    train_acc, test_acc = [], []
    for _ in range(epochs):
        train_metric = train_epoch(model, data_iter, loss_function, optimizer)
        train_acc.append(train_metric)
        with torch.no_grad():
            test_metric = evaluator(model, test_iter)
            test_acc.append(test_metric)
    return train_acc, test_acc


__all__ = ["Accumulator", "accuracy", "evaluator", "train_epoch", "train"]
