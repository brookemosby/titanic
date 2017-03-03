
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

def test_Feature_Engineering():
    """Test function for creating dummies, by checking that the expected number of columns outputted are correct
    """
    trainDF=pd.read_csv('TitanicAttempt/data/train.csv')
    test='TitanicAttempt/data/test.csv'
    train='TitanicAttempt/data/train.csv'
    test=TA.Feature_Engineering(test,trainDF)
    train=TA.Feature_Engineering(train,trainDF)
    assert len(list(train))==16
    assert len(list(test))==15


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