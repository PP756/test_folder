#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:35:27 2024

@author: panagiotispapanastasiou
"""

import numpy as np
import scipy.sparse
from scipy.sparse import save_npz, load_npz
import time
# Multiplication and Addition Rules

def precomputed_addition_table(f):
    """
    Returns the lookup table for addition computations in a specified Galois Field.
    :param f: The Galois Field.
    :return: The precomputed addition table for the specified Galois Field.
    """
    if f == 16:  # GF(2^4)
        table = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                          [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14],
                          [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13],
                          [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12],
                          [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11],
                          [5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8, 11, 10],
                          [6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9],
                          [7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8],
                          [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7],
                          [9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6],
                          [10, 11, 8, 9, 14, 15, 12, 13, 2, 3, 0, 1, 6, 7, 4, 5],
                          [11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4],
                          [12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3],
                          [13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2],
                          [14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1],
                          [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
    else:
        raise ValueError("This Galois field is not currently supported.")
    return table

def precomputed_multiplication_table(f):
    """
    Returns the lookup table for multiplication computations in a specified Galois Field.
    :param f: The Galois Field.
    :return: The precomputed multiplication table for the specified Galois Field.
    """
    if f == 16:  # GF(2^4)
        table = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                          [0, 2, 4, 6, 8, 10, 12, 14, 3, 1, 7, 5, 11, 9, 15, 13],
                          [0, 3, 6, 5, 12, 15, 10, 9, 11, 8, 13, 14, 7, 4, 1, 2],
                          [0, 4, 8, 12, 3, 7, 11, 15, 6, 2, 14, 10, 5, 1, 13, 9],
                          [0, 5, 10, 15, 7, 2, 13, 8, 14, 11, 4, 1, 9, 12, 3, 6],
                          [0, 6, 12, 10, 11, 13, 7, 1, 5, 3, 9, 15, 14, 8, 2, 4],
                          [0, 7, 14, 9, 15, 8, 1, 6, 13, 10, 3, 4, 2, 5, 12, 11],
                          [0, 8, 3, 11, 6, 14, 5, 13, 12, 4, 15, 7, 10, 2, 9, 1],
                          [0, 9, 1, 8, 2, 11, 3, 10, 4, 13, 5, 12, 6, 15, 7, 14],
                          [0, 10, 7, 13, 14, 4, 9, 3, 15, 5, 8, 2, 1, 11, 6, 12],
                          [0, 11, 5, 14, 10, 1, 15, 4, 7, 12, 2, 9, 13, 6, 8, 3],
                          [0, 12, 11, 7, 5, 9, 14, 2, 10, 6, 1, 13, 15, 3, 4, 8],
                          [0, 13, 9, 4, 1, 12, 8, 5, 2, 15, 11, 6, 3, 14, 10, 7],
                          [0, 14, 15, 1, 13, 3, 2, 12, 9, 7, 6, 8, 4, 10, 11, 5],
                          [0, 15, 13, 2, 9, 6, 4, 11, 1, 14, 12, 3, 8, 7, 5, 10]])
    else:
        raise ValueError("This Galois field is not currently supported.")
    return table

tA=precomputed_addition_table(2**4)
tM=precomputed_multiplication_table(2**4)
# =====Load DATA=================
H_sparse=load_npz('140000-0.75-2-8.npz')
or_vector=np.load('raw_stringhomodyne135700.0.npz')['arr_0']
n=H_sparse.shape[1]
vector = np.resize(or_vector, n)

# # =Row Coordinates==============
m=H_sparse.shape[0]
rows = []
c_lil = H_sparse.tolil().astype(dtype=np.uint8).rows
for r in range(m):  # For every row of the VN-to-CN messages array
    rows.append(c_lil[r]) 
  
# =Matrix Values===========
data_type = np.uint32
r = H_sparse.tocoo().row.astype(data_type)
c = H_sparse.tocoo().col.astype(np.uint32)
vals={}
for i in range(len(r)):
    vals[(r[i], c[i])] = H_sparse[r[i], c[i]]
    
    
# Syndrom Calulation (encoding)   
s = np.zeros(m, dtype=np.uint8)  # The highest possible value is 255, which allows a maximum GF of (2^8)
for i in range(0, m):
    for j in range(0, len(rows[i])):
        mul = tM[vals[(i, rows[i][j])]][vector[rows[i][j]]]  # Multiplication step
        s[i] = tA[s[i]][mul]  # Addition step
print(s)
