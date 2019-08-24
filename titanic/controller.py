from titanic.model import TitanicModel
import pandas as pd
import numpy as np

class TitanicController:
    def __init__(self):
        self.m = TitanicModel()
        self._context = './data/'
        self._train = self.create_train()

    def create_train(self) -> object:
        m = self.m
        m.context = self._context
        m.fname = 'train.csv'
        t1 = m.new_dfame()
        print('--------------- train head & column ---------------')
        print(t1.head())
        print(t1.columns)