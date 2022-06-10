import abc
from typing import Callable

import numpy as np


def angular_distance(ra1: np.ndarray, dec1: np.ndarray, ra2: np.ndarray, dec2: np.ndarray):
    s = np.cos(ra1 - ra2) * np.cos(dec1) * np.cos(dec2) + np.sin(dec1) * np.sin(dec2)
    return np.arccos(np.clip(s, -1, 1))


class PDF:
    
    @abc.abstractmethod
    def __call__(self, evt):
        pass


class EnergySignal(PDF):
    
    def __init__(self, spectrum: str, func: Callable = None):
        self.func = func
        self.spectrum = spectrum
        
    def __call__(self, evt):
        return self.func(evt.ra, evt.dec, evt.energy)

    
class AngularSignal(PDF):
    
    def __init__(self, func: Callable = None):
        self.func = func
        
    def __call__(self, evt, ra_src: float, dec_src: float):
        dpsi = angular_distance(evt.ra, evt.dec, ra_src, dec_src)
        return self.func(dpsi, evt.energy)


class EnergyBackground(PDF):
    
    def __init__(self, func: Callable = None):
        self.func = func
    
    def __call__(self, evt):
        return self.func(evt.ra, evt.dec, evt.energy)


class AngularBackground(PDF):
    
    def __init__(self, func: Callable = None):
        self.func = func
    
    def __call__(self, evt):
        return self.func(evt.ra, evt.dec, evt.energy)
