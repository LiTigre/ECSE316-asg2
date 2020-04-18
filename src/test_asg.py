import sys
sys.path.insert(0, '../src')

from my_fourrier import *
from helpers import pad_image
import numpy as np


def test_DFT():
    test_x = np.random.random(2048)
    assert np.allclose(DFT(test_x), np.fft.fft(test_x)) == True, "Arrays should be equal"

def test_FFT():
    test_x = np.random.random(2048)
    assert np.allclose(FFT(test_x), np.fft.fft(test_x)) == True, "Arrays should be equal"

def test_IDFT():
    test_x = np.random.random(2048)
    assert np.allclose(IDFT(DFT(test_x)), np.fft.ifft(np.fft.fft(test_x))) == True, "Arrays should be equal"

def test_IFFT():
    test_x = np.random.random(2048)
    assert np.allclose(IFFT(FFT(test_x)), np.fft.ifft(np.fft.fft(test_x))), "Arrays should be equal"

def test_FFT2():
    test_x = np.random.random((128, 128))
    assert np.allclose(FFT2(test_x), np.fft.fft2(test_x)), "Arrays should be equal"

def test_DFT2():
    test_x = np.random.random((128, 128))
    assert np.allclose(DFT2(test_x), np.fft.fft2(test_x)), "Arrays should be equal"

def test_IFFT2():
    test_x = np.random.random((64, 64))
    assert np.allclose(IFFT2(FFT2(test_x)), np.fft.ifft2(np.fft.fft2(test_x))), "Arrays should be equal"


def test_size_1():
    test_x = np.random.random((5, 24))
    result = pad_image(test_x)
    w, h = result.shape
    assert w % 2 == 0
    assert h % 2 == 0

def test_size_2():
    test_x = np.random.random((4, 24))
    result = pad_image(test_x)
    w, h = result.shape
    assert w % 2 == 0
    assert h % 2 == 0

def test_size_3():
    test_x = np.random.random((5, 32))
    result = pad_image(test_x)
    w, h = result.shape
    assert w % 2 == 0
    assert h % 2 == 0

def test_size_4():
    test_x = np.random.random((64, 32))
    result = pad_image(test_x)
    w, h = result.shape
    assert w % 2 == 0
    assert h % 2 == 0