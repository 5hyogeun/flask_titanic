"""
survival	생존여부	0 = No, 1 = Yes
pclass	승선권	1 = 1st, 2 = 2nd, 3 = 3rd
sex	성별
Age	나이
sibsp	동반한 형제, 자매, 배우자
parch	동반한 부모,자식
ticket	티켓번호
fare	티켓의 요금
cabin	객실번호
embarked	승선한 항구명	C = 쉐부로, Q = 퀸즈타운, S = 사우스햄톤

Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
      dtype='object')

"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics


class TitanicModel:
    def __init__(self):
        self._context = None
        self._fname = None
        self._train = None
        self._test = None
        self._test_id = None

    @property
    def context(self) -> object:return self._context

    @context.setter
    def context(self, context): self._context = context

    @property
    def fname(self) -> object: return self._fname

    @context.setter
    def fname(self, fname): self._fname = fname

    @property
    def train(self) -> object: return self._train

    @context.setter
    def train(self, train): self._train = train

    @property
    def test(self) -> object: return self._test

    @context.setter
    def test(self, test): self._test = test

    @property
    def test_id(self) -> object: return self._test_id

    @context.setter
    def test_id(self, test_id): self._test_id = test_id

    def new_file(self) -> str: return self._context + self._fname

    def new_dfame(self) -> object:
        file = self.new_file()
        return pd.read_csv(file)

    def hook_process(self, train, test) -> object:
        print('----------------1. Cabin Ticket 삭제 -----------------------')
        t = self.drop_feature(train, test, 'Cabin')
        t = self.drop_feature(t[0], t[1], 'Ticket')
        print('----------------2. embarked	승선한 항구명 norminal 편집-----------------------')
        t = self.embarked_norminal(t[0], t[1])
        print('----------------3. Title 편집 -----------------------')
        t = self.title_norminal(t[0], t[1])
        print('----------------4. Name,PassengerId 삭제 -----------------------')
        t = self.drop_feature(t[0], t[1], 'Name')
        self._test_id = test['PassengerId']
        t = self.drop_feature(t[0], t[1], 'PassengerId')
        print('----------------5. Age 편집 -----------------------')
        t = self.age_ordinal(t[0], t[1])
        print('----------------6. Fare ordinal 편집 -----------------------')
        t = self.fare_ordinal(t[0], t[1])
        print('----------------7. Fare 삭제 -----------------------')
        t = self.drop_feature(t[0], t[1], 'Fare')
        print('----------------7. Sex norminal 편집 -----------------------')
        t = self.sex_norminal(t[0], t[1])
        t[1] = t[1].fillna({"FareBand": 1})
        a = self.null_sum(t[1])
        print('널의 수량 {} 개'.format(a))
        self._test = t[1]
        return t[0]

    @staticmethod
    def null_sum(train) -> int:
        return train.isnull().sum()

    @staticmethod
    def drop_feature(train, test, feature) ->[]:
        train = train.drop([feature], axis = 1)
        test = test.drop([feature], axis = 1)
        return [train, test]

    @staticmethod
    def embarked_norminal(train, test) -> []:
        print('>>>> embarked_norminal')
        # c_city = train[train['Embarked'] == 'C'].shape[0]
        # s_city = train[train['Embarked'] == 'S'].shape[0]
        # q_city = train[train['Embarked'] == 'Q'].shape[0]

        train = train.fillna({"Embarked" : "S"})
        city_mapping = {"S": 1, "C": 2, "Q": 3}
        train['Embarked'] = train['Embarked'].map(city_mapping)
        test['Embarked'] = test['Embarked'].map(city_mapping)
        return [train, test]

    @staticmethod
    def title_norminal(train, test) -> []:
        print('>>>> title_norminal')
        combine = [train, test]
        for dataset in combine:
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)

        for dataset in combine:
            dataset['Title'] \
                = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major','Rev', 'Jonkheer', 'Dona'], 'Rare')
            dataset['Title'] \
                = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            dataset['Title'] \
                = dataset['Title'].replace(['Mile', 'Ms'], 'Miss')

        train[['Title','Survived']].groupby(['Title'], as_index=False).mean()
        # print(train[['Title','Survived']].groupby(['Title'], as_index=False).mean())
        title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Royal':5, 'Rare':6, 'Mne':7}
        for dataset in combine:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0)
        return [train, test]

    @staticmethod
    def sex_norminal(train, test) -> []:
        combine  = [train, test]
        sex_mapping = {'male':0, 'female':1}
        for dataset in combine:
            dataset['Sex'] = dataset['Sex'].map(sex_mapping)

        return [train, test]

    @staticmethod
    def age_ordinal(train, test)->[]:
        train['Age'] = train['Age'].fillna(-0.5)
        test['Age'] = test['Age'].fillna(-0.5)
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
        labels = ['Unknown', 'Baby', 'Child', 'Teenager','Student','Young Adult','Adult','Senior']
        train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        test['AgeGroup'] = pd.cut(test['Age'], bins, labels=labels)
        age_title_mapping = {0: 'Unknown', 1: 'Baby', 2: 'Child',
                             3:'Teenager', 4:'Student', 5: 'Young Adult', 6:'Adult', 7:'Senior'}
        for x in range(len(train['AgeGroup'])):
            if train['AgeGroup'][x] == 'Unknown':
                train['AgeGroup'][x] = age_title_mapping[train['Title'][x]]
        for x in range(len(test['AgeGroup'])):
            if test['AgeGroup'][x] == 'Unknown':
                test['AgeGroup'][x] = age_title_mapping[test['Title'][x]]

        age_mapping = {'Unknown':0, 'Baby':1, 'Child':2,
                             'Teenager':3, 'Student':4, 'Young Adult':5, 'Adult':6, 'Senior':7}

        train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
        test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
        print(train['AgeGroup'].head())
        return [train, test]

    @staticmethod
    def fare_ordinal(train, test) ->[]:
        train['FareBand'] = pd.qcut(train['Fare'], 4, labels={1,2,3,4})
        test['FareBand'] = pd.qcut(test['Fare'], 4, labels={1, 2, 3, 4})
        return [train, test]

    # 검증 알고리즘 작성

    def hook_test(self, model, dummy):
        print('KNN 활용한 검증 정확도 {} %'.format(self.accuracy_by_knn(model, dummy)))   #  1
        print('결정트리 활용한 검증 정확도 {} %'.format(self.accuracy_by_dtree(model, dummy)))   #  2
        print('랜덤포리스트 활용한 검증 정확도 {} %'.format(self.accuracy_by_rforest(model, dummy)))    #  3
        print('나이브베이즈 활용한 검증 정확도 {} %'.format(self.accuracy_by_nb(model, dummy)))    #  4
        print('SVM 활용한 검증 정확도 {} %'.format(self.accuracy_by_svm(model, dummy)))   #  5

    @staticmethod
    def create_k_fold():
        k_fold = KFold(n_splits=10, shuffle=True, random_state=0)
        return k_fold

    @staticmethod
    def create_random_variables(train, X_feature, Y_features) -> []:
        the_X_feature = X_feature
        the_Y_feature = Y_features
        train2, test2 = train_test_split(train, test_size=0.3, random_state=0)
        train_X = train2[the_X_feature]
        train_Y = train2[the_Y_feature]
        test_X = test2[the_X_feature]
        test_Y = test2[the_Y_feature]
        return [train_X, train_Y, test_X, test_Y]

    def accuracy_by_knn(self, model, dummy):
        clf = KNeighborsClassifier(n_neighbors=13)
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy =round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_dtree(self, model, dummy):
        clf = DecisionTreeClassifier()
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_rforest(self, model, dummy):
        clf = RandomForestClassifier(n_estimators=13)
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_nb(self, model, dummy):
        clf = GaussianNB()
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_svm(self, model, dummy):
        clf = SVC()
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy