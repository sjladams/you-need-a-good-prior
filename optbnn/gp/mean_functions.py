# Copyright 2016 James Hensman, alexggmatthews, PabloLeon, Valentine Svensson
# Copyright 2017 Thomas Viehmann
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
from . import parameter

CONS_2PI = 2 * math.pi

class MeanFunction(torch.nn.Module):
    """
    The base mean function class.
    To implement a mean function, write the __call__ method. This takes a
    tensor X and returns a tensor m(X). In accordance with the GPflow
    standard, each row of X represents one datum, and each row of Y is computed
    independently for each row of X.

    MeanFunction classes can have parameters, see the Linear class for an
    example.
    """
    def forward(self, X):
        raise NotImplementedError("Implement the forward method for this mean function")

    def __add__(self, other):
        return Additive(self, other)

    def __mul__(self, other):
        return Product(self, other)


class Zero(MeanFunction):
    def forward(self, X):
        return torch.zeros(X.size(0), 1, dtype=X.dtype, device=X.device)


class Linear(MeanFunction):
    """
    y_i = A x_i + b
    """
    def __init__(self, A=None, b=None):
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.

        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be D x Q, b must be a vector of length Q.
        """
        A = torch.ones((1, 1)) if A is None else A
        b = torch.zeros(1) if b is None else b
        MeanFunction.__init__(self)
        if A.dim()==1:
            A = A.unsqueeze(1)
        self.A = parameter.Param(A)
        self.b = parameter.Param(b)

    def __call__(self, X):
        return torch.matmul(X, self.A.get()) + self.b.get()


class Sine(MeanFunction):
    """
    y_i = sin(x_i)
    """
    def __init__(self, amp: float = 1., freq: float = 1., phase: float = 0.):
        MeanFunction.__init__(self)

        self.amp = parameter.Param(amp)
        self.freq = parameter.Param(freq)
        self.phase = parameter.Param(phase)

    def __call__(self, X):
        return self.amp.get() * torch.sin(CONS_2PI * self.freq.get() * X + self.phase.get())


class Additive(MeanFunction):
    def __init__(self, first_part, second_part) -> None:
        MeanFunction.__init__(self)
        self.add_1 = first_part
        self.add_2 = second_part

    def __call__(self, X):
        return torch.add(self.add_1(X), self.add_2(X))


class Product(MeanFunction):
    def __init__(self, first_part, second_part):
        MeanFunction.__init__(self)

        self.prod_1 = first_part
        self.prod_2 = second_part

    def __call__(self, X):
        return torch.multiply(self.prod_1(X), self.prod_2(X))
