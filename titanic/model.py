"""
survival	생존여부	0 = No, 1 = Yes
pclass	승선권	1 = 1st, 2 = 2nd, 3 = 3rd
sex	성별
Age	나이 in years
sibsp	동반한 형제, 자매, 배우자
parch	동반한 부모, 자식
ticket	티켓번호
fare	티켓의 요금
cabin	객실번호
embarked	승선한 항구명	C = 쉐부로, Q = 퀸즈타운, S = 사우스햄튼

변수(variable)는 곧 feature
"""
import pandas as pd
import numpy as np

class TitanicModel:
    def __init__(self):
        self._context = None
        self._fname = None
        self._train = None
        self._test = None
        self._test_id = None

    @property
    def context(self) -> object:return self._context    #read only

    @context.setter
    def context(self, context): self._context = context

    @property
    def fname(self) -> object: return self._fname  # read only

    @context.setter
    def fname(self, fname): self._fname  = fname

    @property
    def train(self) -> object: return self._train  # read only

    @context.setter
    def train(self, train): self._train = train

    @property
    def test(self) -> object: return self._test  # read only

    @context.setter
    def test(self, test): self._test = test

    @property
    def test_id(self) -> object: return self._test_id  # read only

    @context.setter
    def test_id(self, test_id): self._test_id = test_id

    def new_file(self) -> str: return self._context + self._fname
# method
    def new_dfame(self) -> object:
        file = self.new_file()
        return pd.read_csv(file)

    def hook_process(self, train, test) -> object:     # 상위 개념
        print('----------------------1. ----------------------')
        print('----------------------2. ----------------------')
        print('----------------------3. ----------------------')

