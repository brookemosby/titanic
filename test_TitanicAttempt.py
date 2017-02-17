
import TitanicAttempt as TA
import sklearn
import pandas as pd
def test_Read_Files():
    train='train.csv'
    test='test.csv'
    train=TA.Read_Files(train)
    test=TA.Read_Files(test)
    assert train['PassengerId'].min()==1
    assert train['PassengerId'].max()==891
    assert test['PassengerId'].min()==892
    assert test['PassengerId'].max()==1309
               
def test_Changing_Name_to_Titles():
    train='train.csv'
    test='test.csv'
    train=TA.Changing_Name_to_Titles(train)
    test=TA.Changing_Name_to_Titles(test)
    assert len(list(train))==13
    assert len(list(test))==12
              
def test_Creating_NewCols():
    train='train.csv'
    test='test.csv'
    train=TA.Creating_NewCols(train)
    test=TA.Creating_NewCols(test)
    assert len(list(train))==12
    assert len(list(test))==11

def test_Getting_Dummies():
    trainDF=pd.read_csv('train.csv')
    test='test.csv'
    train='train.csv'
    test=TA.Getting_Dummies(test,trainDF)
    train=TA.Getting_Dummies(train,trainDF)
    assert len(list(train))==15
    assert len(list(test))==14


def test_Create_Decision_Tree():
    x=TA.Create_Decision_Tree()
    assert type(x)==sklearn.tree.tree.DecisionTreeClassifier
               
def test_Produce_Predictions():
    TA.Produce_Predictions('TestFunc.csv')
    assert not pd.read_csv('TestFunc.csv').empty