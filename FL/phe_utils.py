

import phe.paillier as paillier
from joblib import Parallel, delayed
import numpy as np


class DigitEncrypt(object):
    def __init__(self, plaintext, public_key, encrypted=False):
        self.public_key = public_key
        self.ciphertext = self.encrypt(plaintext) if not encrypted else plaintext

    def encrypt(self, plaintext):
        ciphertext = self.public_key.encrypt(plaintext)
        return ciphertext

    def decrypt(self, private_key):
        plaintext = private_key.decrypt(self.ciphertext)
        return plaintext
    
    def getciphertext(self):
        return self.ciphertext
    
    def getplaintext(self, private_key):
        return self.decrypt(private_key)

    def __add__(self, other):
        return DigitEncrypt(self.getciphertext()+other.getciphertext(), self.public_key, True)


class Array2DEncrypt(DigitEncrypt):
    def __init__(self, plaintext, public_key, encrypted=False):
        super().__init__(plaintext, public_key, encrypted)

    def encrypt(self, plaintext):
        ciphertext = list(Parallel(n_jobs=-1)(delayed(
            # lambda x: self.public_key.encrypt(x)
            lambda x: DigitEncrypt(x, self.public_key)
        )(p) for p in plaintext))
        return ciphertext

    def decrypt(self, private_key):
        plaintext = list(Parallel(n_jobs=-1)(delayed(
            lambda c: c.getplaintext(private_key)
        )(c) for c in self.ciphertext))
        return plaintext

    def __add__(self, other):
        if (len(self.getciphertext()) != len(other.getciphertext())):
            raise Exception
        addlist = [a+b for a, b in zip(self.getciphertext(), other.getciphertext())]
        return Array2DEncrypt(addlist, self.public_key, True)


class Array3DEncrypt(Array2DEncrypt):

    def encrypt(self, plaintext):
        ciphertext = list(Parallel(n_jobs=-1)(delayed(
            lambda x: Array2DEncrypt(x, self.public_key)
        )(p) for p in plaintext))
        return ciphertext

    def decrypt(self, private_key):
        plaintext = list(Parallel(n_jobs=-1)(delayed(
            lambda c: c.getplaintext(private_key)
        )(c) for c in self.ciphertext))
        return plaintext

    def __add__(self, other):
        # if (len(self.getciphertext()) != len(other.getciphertext())):
        #     raise Exception
        addlist = [a+b for a, b in zip(self.getciphertext(), other.getciphertext())]
        return Array3DEncrypt(addlist, self.public_key, True)
        

def demo():
    public_key, private_key = paillier.generate_paillier_keypair()

    import random
    import time

    ## test digit
    a = DigitEncrypt(1, public_key)
    b = DigitEncrypt(2, public_key)
    c = a + b
    print(c.getplaintext(private_key))


    ## test array
    a = Array2DEncrypt([1, 2, 3], public_key)
    b = Array2DEncrypt([2, 3, 1], public_key)
    c = a + b
    print(c.getplaintext(private_key))


    # test tensor
    a = Array3DEncrypt([[1, 2, 3], [3, 2, 1]], public_key)
    b = Array3DEncrypt([[2, 3, 1], [1, 3, 2]], public_key)
    c = a + b
    print(c.getplaintext(private_key))


demo()