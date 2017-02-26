
from TitanicAttempt import TitanicAttempt as TA
import sklearn
import pandas as pd
def test_Read_Files():
    """Test function for reading in files, checks that expected amount of Passengers are correct for each DataFrame
    """
    train='TitanicAttempt/data/train.csv'
    test='TitanicAttempt/data/test.csv'
    train=TA.Read_Files(train)
    test=TA.Read_Files(test)
    assert train['PassengerId'].min()==1
    assert train['PassengerId'].max()==891
    assert test['PassengerId'].min()==892
    assert test['PassengerId'].max()==1309
               
def test_Changing_Name_to_Titles():
    """Test function for creating 'Title' columns, by checking that the expected number of columns outputted are correct
    """
    train='TitanicAttempt/data/train.csv'
    test='TitanicAttempt/data/test.csv'
    train=TA.Changing_Name_to_Titles(train)
    test=TA.Changing_Name_to_Titles(test)
    assert len(list(train))==13
    assert len(list(test))==12
              
def test_Creating_NewCols():
    """Test function for creating columns, by checking that the expected number of columns outputted are correct
    """
    train='TitanicAttempt/data/train.csv'
    test='TitanicAttempt/data/test.csv'
    train=TA.Creating_NewCols(train)
    test=TA.Creating_NewCols(test)
    assert len(list(train))==12
    assert len(list(test))==11

def test_Getting_Dummies():
    """Test function for creating dummies, by checking that the expected number of columns outputted are correct
    """
    trainDF=pd.read_csv('TitanicAttempt/data/train.csv')
    test='TitanicAttempt/data/test.csv'
    train='TitanicAttempt/data/train.csv'
    test=TA.Getting_Dummies(test,trainDF)
    train=TA.Getting_Dummies(train,trainDF)
    assert len(list(train))==15
    assert len(list(test))==14


def test_Create_Random_Forest():
    """Checks that the type of return item from Create_Random_Forest() is in fact a Random Forest
    """
    x=TA.Create_Random_Forest('TitanicAttempt/data/train.csv')
    assert type(x)==sklearn.ensemble.RandomForestClassifier
               
def test_Produce_Predictions():
    """Checks that the prediction file created from Produce_Predictions is not empty
    """
    TA.Produce_Predictions('TestFunc.csv','TitanicAttempt/data/train.csv','TitanicAttempt/data/test.csv')
    assert not pd.read_csv('TestFunc.csv').empty