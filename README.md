# Micrograd From Scratch

A small automatic differentiation engine and neural network library built from scratch in Python, inspired by Andrej Karpathy's Micrograd lecture.

The goal of this project is to understand what happens underneath modern deep learning frameworks such as PyTorch, rather than simply using them as black boxes.

## What This Project Covers

* Custom `Value` objects
* Operator overloading
* Computational graphs
* Derivatives
* Chain rule
* Backpropagation
* Automatic differentiation
* Gradients
* Basic neural networks
* Training a neural network using gradient descent

## Why I Built This

Modern deep learning frameworks provide automatic differentiation and neural-network components out of the box.

Instead of only using these tools, this project implements the core ideas from scratch to understand:

```text
Forward Pass
     ↓
Computational Graph
     ↓
Loss
     ↓
Derivatives
     ↓
Backpropagation
     ↓
Gradient Descent
     ↓
Updated Parameters
```

## Example

```python
from micrograd import Value

a = Value(3)
b = Value(5)

c = a + b

print(c)
```

The `Value` class stores not only the numerical value, but also information required to construct the computational graph and calculate gradients.

Operator overloading allows expressions such as:

```python
y = a * b + c
```

to work naturally with custom `Value` objects.

## Project Structure

```text
micrograd-from-scratch/
│
├── micrograd/
│   └── engine.py
│
├── demo.ipynb
│
├── README.md
│
└── requirements.txt
```

## Learning Focus

This project is primarily an educational implementation.

The main focus is understanding the mechanics behind automatic differentiation and neural-network training:

* How computational graphs are created
* How local derivatives are calculated
* How the chain rule connects derivatives
* How gradients flow backward through a graph
* How gradients are used to update model parameters

## Inspiration

Inspired by Andrej Karpathy's educational material on building neural networks from scratch.

## Status

Work in progress — additional functionality and experiments will be added as I progress through the implementation.
