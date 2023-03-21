'''
Author: ZJG
Date: 2022-11-08 22:21:51
LastEditors: ZJG
LastEditTime: 2023-03-04 11:16:06
'''

import itertools
import logging
import math
import os
import unittest

import crypten
import crypten.communicator as comm
import torch
import torch.nn.functional as F
from crypten.common.functions.pooling import _pool2d_reshape
from crypten.common.rng import generate_kbit_random_tensor, generate_random_ring_element
from crypten.common.tensor_types import is_float_tensor
from crypten.config import cfg
from crypten.mpc import MPCTensor, ptype as Ptype
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor
from test.multiprocess_test_case import get_random_test_tensor, MultiProcessTestCase
import time
import crypten.mpc as mpc
from crypten.mpc.primitives.arrayAccess import SecureArrayAccess

# os.environ['GLOO_SOCKET_IFNAME'] = 'eth1'
device = 'cpu'
def _get_random_test_tensor(*args, **kwargs):
        return get_random_test_tensor(device=device, *args, **kwargs)


def test_array_access():
    """Test embedding on encrypted tensor."""
    matrix_size = (50000, 512)
    matrix = _get_random_test_tensor(size=matrix_size, is_float=True)
    # encrypted_matrix = crypten.cryptensor(matrix)
    # index = crypten.cryptensor([1], precision=0)
    # print(matrix.size())
    t = time.perf_counter()
    for i in range(10):
        result = SecureArrayAccess(matrix, 1)
    cost = time.perf_counter() - t
    print("Cost:"+str(cost))
    
def test_embedding():
    matrix_size = (50000, 512)
    matrix = _get_random_test_tensor(size=matrix_size, is_float=True)
    encrypted_matrix = MPCTensor(matrix)
    # print(encrypted_matrix)
    t = time.perf_counter()
    for i in range(10):
        list = [0.]*49999
        list.append(1.)
        hot = torch.tensor(list)
        hot_enc = crypten.cryptensor(hot)
        result = hot_enc.matmul(encrypted_matrix)
    cost = time.perf_counter() - t
    print("Cost:"+str(cost))
# @mpc.run_multiprocess(world_size=3)  
def test_relu():
        """Test relu on encrypted tensor."""
        # for width in range(2, 5):
        matrix_size = (5, 5)
        matrix = _get_random_test_tensor(size=matrix_size, is_float=True)

        # Generate some negative values
        matrix2 = _get_random_test_tensor(size=matrix_size, is_float=True)
        matrix = matrix - matrix2

        encrypted_matrix = MPCTensor(matrix)
        reference = F.relu_(matrix)
        t = time.perf_counter()
        for i in range(1000):
            encrypted_matrix = encrypted_matrix.relu()
        cost = time.perf_counter() - t
        print("Cost:"+str(cost))


# @mpc.run_multiprocess(world_size=3)    
def test_broadcast_arithmetic_ops():
        """Test broadcast of arithmetic functions."""
        arithmetic_functions = ["add", "sub", "mul", "div"]
        # TODO: Add broadcasting for pos_pow since it can take a tensor argument
        arithmetic_sizes = [
            (),
            (1,),
            (2,),
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (1, 1, 1),
            (1, 1, 2),
            (1, 2, 1),
            (2, 1, 1),
            (2, 2, 2),
            (1, 1, 1, 1),
            (1, 1, 1, 2),
            (1, 1, 2, 1),
            (1, 2, 1, 1),
            (2, 1, 1, 1),
            (2, 2, 2, 2),
        ]
        tensor_type = MPCTensor
        func = "div"
        # for tensor_type in [lambda x: x, MPCTensor]:
        #     for func in arithmetic_functions:
        # for size1, size2 in itertools.combinations(arithmetic_sizes, 2):
        size1 = (1, 1, 1, 1)
        size2 = (1, 1, 1, 2)
        exclude_zero = True if func == "div" else False
        # multiply denominator by 10 to avoid dividing by small num
        const = 10 if func == "div" else 1

        tensor1 = _get_random_test_tensor(size=size1, is_float=True)
        tensor2 = _get_random_test_tensor(
            size=size2, is_float=True, ex_zero=exclude_zero
        )
        tensor2 *= const
        encrypted1 = MPCTensor(tensor1)
        encrypted2 = tensor_type(tensor2)
        reference = getattr(tensor1, func)(tensor2)
        t = time.perf_counter()
        for i in range(1000):
            encrypted_out = getattr(encrypted1, func)(encrypted2)
        cost = time.perf_counter() - t
        print("Cost:"+str(cost))

        # private = isinstance(encrypted2, MPCTensor)
        # self._check(
        #     encrypted_out,
        #     reference,
        #     "%s %s broadcast failed"
        #     % ("private" if private else "public", func),
        # )

        # Test with integer tensor
        # tensor2 = self._get_random_test_tensor(
        #     size=size2, is_float=False, ex_zero=exclude_zero
        # )
        # tensor2 *= const
        # reference = getattr(tensor1, func)(tensor2.float())
        # encrypted_out = getattr(encrypted1, func)(tensor2)
        # self._check(
        #     encrypted_out,
        #     reference,
        #     "%s broadcast failed with public integer tensor" % func,
        # )
                    
# @mpc.run_multiprocess(world_size=3) 
def test_comparators():
    """Test comparators (>, >=, <, <=, ==, !=)"""
    # ["gt", "ge", "lt", "le", "eq", "ne"]
    comp = "gt"
    # for tensor_type in [lambda x: x, MPCTensor]:
    tensor_type = MPCTensor
    tensor1 = _get_random_test_tensor(is_float=True)
    tensor2 = _get_random_test_tensor(is_float=True)

    encrypted_tensor1 = MPCTensor(tensor1)
    encrypted_tensor2 = tensor_type(tensor2)
    reference = getattr(tensor1, comp)(tensor2).float()
    t = time.perf_counter()
    for i in range(1000):
        encrypted_out = getattr(encrypted_tensor1, comp)(encrypted_tensor2)
    cost = time.perf_counter() - t
    print(tensor_type)
    print("     reference"+str(reference))
    print("     encrypted_out"+str(encrypted_out.get_plain_text()))
    print("     encrypted_out"+str(encrypted_out))
    print("Cost:"+str(cost))

    # _check(encrypted_out, reference, "%s comparator failed" % comp)

    # Check deterministic example to guarantee all combinations
    # tensor1 = torch.tensor([2.0, 3.0, 1.0, 2.0, 2.0])
    # tensor2 = torch.tensor([2.0, 2.0, 2.0, 3.0, 1.0])

    # encrypted_tensor1 = MPCTensor(tensor1)
    # encrypted_tensor2 = tensor_type(tensor2)

    # reference = getattr(tensor1, comp)(tensor2).float()
    # t = time.perf_counter()
    # for i in range(1000):
    #     encrypted_out = getattr(encrypted_tensor1, comp)(encrypted_tensor2)
    # cost = time.perf_counter() - t
    # print("Cost:"+str(cost))

    # self._check(encrypted_out, reference, "%s comparator failed" % comp)
# @mpc.run_multiprocess(world_size=3)           
def test_ltz():
    """Test comparators (>, >=, <, <=, ==, !=)"""
    comp = "gt"
    # for tensor_type in [lambda x: x, MPCTensor]:
    tensor_type = MPCTensor
    tensor = _get_random_test_tensor(is_float=True)

    encrypted_tensor = MPCTensor(tensor)
    # reference = getattr(tensor1, comp)(tensor2).float()
    t = time.perf_counter()
    for i in range(1000):
        encrypted_out = encrypted_tensor._ltz()
    cost = time.perf_counter() - t
    print("     encrypted_out"+str(encrypted_out.get_plain_text()))
    print("     encrypted_out"+str(encrypted_out))
    print("Cost:"+str(cost))
    
    
if __name__ == "__main__":
    crypten.init()
    # print(os.environ)
    test_array_access()
    # test_embedding()
   # test_ltz()
    #test_comparators()
    #test_relu()
