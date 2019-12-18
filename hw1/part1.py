#!/usr/bin/env python3
"""
part1.py

UNSW COMP9444 Neural Networks and Deep Learning

ONLY COMPLETE METHODS AND CLASSES MARKED "TODO".

DO NOT MODIFY IMPORTS. DO NOT ADD EXTRA FUNCTIONS.
DO NOT MODIFY EXISTING FUNCTION SIGNATURES.
DO NOT IMPORT ADDITIONAL LIBRARIES.
DOING SO MAY CAUSE YOUR CODE TO FAIL AUTOMATED TESTING.
"""
import torch

# Simple addition operation

def simple_addition(x, y):
    """
    TODO: Implement a simple addition function that accepts two tensors and returns the result.
    """
    if torch.is_tensor(x) and torch.is_tensor(y):
        torch_add = torch.add(x,y)
        return torch_add
    else:
        return False

# Resize tensors
# Use view() to implement the following functions ( flatten() and reshape() are not allowed )

def simple_reshape(x, shape):
    """
    TODO: Implement a function that reshapes the given tensor as the given shape and returns the result.
    """
    if torch.is_tensor(x):
        torch_reshaped = x.view(shape)
        return torch_reshaped
    else:
        return False

def simple_flat(x):
    """
    TODO: Implement a function that flattens the given tensor and returns the result.
    """
    if torch.is_tensor(x):
        torch_flattened = torch.flatten(x)
        return torch_flattened
    else:
        return False


# Transpose and Permutation

def simple_transpose(x):
    """
    TODO: Implement a function that swaps the first dimension and
        the second dimension of the given matrix x and returns the result.
    """
    if torch.is_tensor(x):
        torch_transposed = torch.transpose(x,0,1)
        return torch_transposed
    else:
        return False


def simple_permute(x, order):
    """
    TODO: Implement a function that permute the dimensions of the given tensor
        x according to the given order and returns the result.
    """
    if torch.is_tensor(x):
        torch_permuted = x.permute(order)
        return torch_permuted
    else:
        return False

# Matrix multiplication (with broadcasting).

def simple_dot_product(x, y):
    """
    TODO: Implement a function that computes the dot product of
        two rank 1 tensors and returns the result.
    """
    if torch.is_tensor(x) and torch.is_tensor(y):
        dot_product = torch.dot(x,y)
        return dot_product
    else:
        return False

def simple_matrix_mul(x, y):
    """
    TODO: Implement a function that performs a matrix multiplication
        of two given rank 2 tensors and returns the result.
    """
    if torch.is_tensor(x) and torch.is_tensor(y):
        matmul_product = torch.mm(x,y)
        return matmul_product
    else:
        return False

def broadcastable_matrix_mul(x, y):
    """
    TODO: Implement a function that computes the matrix product of two tensors and returns the result.
        The function needs to be broadcastable.
    """
    if torch.is_tensor(x) and torch.is_tensor(y):
        b_matmul_product = torch.matmul(x,y)
        return b_matmul_product
    else:
        return False

# Concatenate and stack.
def simple_concatenate(tensors):
    """
    TODO: Implement a function that concatenates the given sequence of tensors
        in the first dimension and returns the result
    """
    torch_concatenation = torch.cat(tensors,0)
    return torch_concatenation

def simple_stack(tensors, dim):
    """
    TODO: Implement a function that concatenates the given sequence of tensors
        along a new dimension(dim) and returns the result.
    """
    torch_stack = torch.stack(tensors, dim)
    return torch_stack

