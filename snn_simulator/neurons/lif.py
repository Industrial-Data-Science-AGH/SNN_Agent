import numpy as np


class LIFNeuron:
    def __init__(self, beta, threshold):
        self.beta = beta
        self.threshold = threshold
        self.V = 0

    def reset(self):
        self.V = 0

    def step(self, I):
        self.V = self.beta * self.V + I

        spike = 0
        if self.V >= self.threshold:
            spike = 1
            self.V = 0

        return spike
