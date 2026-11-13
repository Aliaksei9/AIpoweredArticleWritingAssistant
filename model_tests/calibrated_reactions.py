"""A single OOF-fitted calibration coefficient for reaction forecasts."""
import numpy as np


class CalibratedReactionHead:
    def __init__(self, base_head, coefficient):self.base_head=base_head;self.coefficient=float(coefficient)

    def predict(self, rows, vectors, texts):
        prior=self.base_head.model._prior(rows)[:,1]
        value=self.base_head.predict(rows,vectors,texts)
        return np.rint(np.maximum(0,prior+self.coefficient*(value-prior)))


def calibration_coefficient(target, prior, prediction):
    delta=prediction-prior;denominator=float(delta@delta)
    return float(np.clip(delta@(target-prior)/denominator,0.,3.)) if denominator>1e-12 else 1.
