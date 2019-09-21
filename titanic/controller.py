from titanic.model import TitanicModel
import pandas as pd
import numpy as np
class TitanicController:
    def __init__(self):
        self._m = TitanicModel()
        self._context = './data/'
        self._train = self.create_train()

    def create_train(self)->object:
        print('0 >>>> ')
        m = self._m
        print('1 >>>> ' + self._context)
        m.context = self._context
        print('2 >>>> '+m.context)
        m.fname = 'train.csv'
        t1 = m.new_dfame()

        m.fname = 'test.csv'
        t2 = m.new_dfame()

        return m.hook_process(t1, t2)

