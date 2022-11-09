'''
Author: ZJG
Date: 2022-11-08 22:21:51
LastEditors: ZJG
LastEditTime: 2022-11-09 10:23:53
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


device = 'cpu'
def _get_random_test_tensor(*args, **kwargs):
        return get_random_test_tensor(device=device, *args, **kwargs)
    
    
def test_comparators():
    """Test comparators (>, >=, <, <=, ==, !=)"""
    # ["gt", "ge", "lt", "le", "eq", "ne"]
    for comp in ["gt"]:
        for tensor_type in [lambda x: x, MPCTensor]:
            tensor1 = _get_random_test_tensor(is_float=True)
            tensor2 = _get_random_test_tensor(is_float=True)

            encrypted_tensor1 = MPCTensor(tensor1)
            encrypted_tensor2 = tensor_type(tensor2)
            reference = getattr(tensor1, comp)(tensor2).float()
            encrypted_out = getattr(encrypted_tensor1, comp)(encrypted_tensor2)
            print(tensor_type)
            print("     reference"+str(reference))
            print("     encrypted_out"+str(encrypted_out.get_plain_text()))

            # _check(encrypted_out, reference, "%s comparator failed" % comp)

            # Check deterministic example to guarantee all combinations
            # tensor1 = torch.tensor([2.0, 3.0, 1.0, 2.0, 2.0])
            # tensor2 = torch.tensor([2.0, 2.0, 2.0, 3.0, 1.0])

            # encrypted_tensor1 = MPCTensor(tensor1)
            # encrypted_tensor2 = tensor_type(tensor2)

            # reference = getattr(tensor1, comp)(tensor2).float()
            # encrypted_out = getattr(encrypted_tensor1, comp)(encrypted_tensor2)

            # self._check(encrypted_out, reference, "%s comparator failed" % comp)
            
if __name__ == "__main__":
    crypten.init()
    test_comparators()