

import phe.paillier as paillier
from joblib import Parallel, delayed
import numpy as np
import numbers


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
    
    def __mul__(self, c):
        assert(isinstance(c, numbers.Number))
        return DigitEncrypt(self.getciphertext()*c, self.public_key, True)

class ArrayEncrypt(DigitEncrypt):
    def __init__(self, plaintext, public_key, encrypted=False):
        super().__init__(plaintext, public_key, encrypted)

    def encrypt(self, plaintext):
        def _encrypt_helper(x):
            return ArrayEncrypt(x, self.public_key) if isinstance(x, list) else DigitEncrypt(x, self.public_key)
        ciphertext = list(Parallel(n_jobs=-1)(delayed(_encrypt_helper)(p) for p in plaintext))
        return ciphertext

    def decrypt(self, private_key):
        if isinstance(self.ciphertext, list):
            plaintext = list(Parallel(n_jobs=-1)(delayed(
                lambda c: c.getplaintext(private_key)
            )(c) for c in self.ciphertext))
        else:
            plaintext = self.ciphertext.getplaintext(private_key)
        return plaintext

    def __add__(self, other):
        if (len(self.getciphertext()) != len(other.getciphertext())):
            raise Exception
        assert type(self) == type(other)
        addlist = [a+b for a, b in zip(self.getciphertext(), other.getciphertext())]
        return ArrayEncrypt(addlist, self.public_key, True)
    
    def __mul__(self, c):
        assert(isinstance(c, numbers.Number))
        mullist = [a*c for a in self.getciphertext()]
        return ArrayEncrypt(mullist, self.public_key, True)


def demo():
    public_key, private_key = paillier.generate_paillier_keypair()

    import random
    import time

    a = DigitEncrypt(1, public_key)
    ## test digit
    a = DigitEncrypt(1, public_key)
    b = DigitEncrypt(2, public_key)
    c = a + b
    print(c.getplaintext(private_key))


    ## test array
    a = ArrayEncrypt([1, 2, 3], public_key)
    b = ArrayEncrypt([2, 3, 1], public_key)
    c = a*2.2 + b*1
    print(c.getplaintext(private_key))


    # test tensor
    pa = np.random.randn(2, 2, 2, 2)
    pb = np.random.randn(2, 2, 2, 2)
    a = ArrayEncrypt(pa.tolist(), public_key)
    b = ArrayEncrypt(pb.tolist(), public_key)
    c = a + b
    print(c.getplaintext(private_key) == (pa + pb).tolist())


# demo()