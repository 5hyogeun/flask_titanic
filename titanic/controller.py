from titanic.model import TitanicModel
import pandas as pd
import numpy as np
from sklearn.svm import SVC
class TitanicController:
    def __init__(self):
        self._m = TitanicModel()
        print('***********************************')
        print(self._m.test_id)
        ##self._context = './data/'
        self._train = self.create_train()

    def create_train(self)->object:
        print('0 >>>> ')
        m = self._m
        # print('1 >>>> ' + self._context)
        m.context = './data/'
        print('2 >>>> '+m.context)
        m.fname = 'train.csv'
        t1 = m.new_dfame()

        m.fname = 'test.csv'
        t2 = m.new_dfame()

        return m.hook_process(t1, t2)

    def create_model(self) -> object:
        train = self._train
        model = train.drop('Survived', axis = 1)
        print('---- Model Info ----')
        print(model.info)
        return model

    def create_dummy(self) -> object:
        train = self._train
        dummy = train['Survived']
        return dummy

    def test_all(self):
        model = self.create_model()
        dummy = self.create_dummy()
        m = self._m
        m.hook_test(model, dummy)

    def submit(self):
        m = self._m
        model = self.create_model()
        dummy = self.create_dummy()
        test = m._test
        test_id = m._test_id     #######################  --주의 --  #######################

        clf = SVC()
        clf.fit(model, dummy)
        print('---- 1. Model Info ----')
        print(model.info)
        print('---- 2. ----')
        prediction = clf.predict(test)
        print('---- 3. ----')
        submission = pd.DataFrame(
            {'PassengerId':test_id, 'Survived': prediction})
        print('submission.head()')
        print(submission.head())
        submission.to_csv('./data/submission.csv', index=False)