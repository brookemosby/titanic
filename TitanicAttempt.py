import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC

def Read_Files(FileName):

    DataFrame=pd.read_csv(FileName)
    return DataFrame


def Changing_Name_to_Titles(DataFrame):
    
    DataFrame= Read_Files(DataFrame)
    titles=DataFrame['Name'].apply(lambda x: x.split(',')[1].split(' ')[1])
    title_mapping = {"the":7, "Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Dr.": 5, "Rev.": 6, "Major.": 7, "Col.": 7, "Mlle.": 8, "Mme.": 8, "Don.": 9, "Lady.": 10, "Countess.": 10, "Jonkheer.": 10, "Sir.": 9, "Capt.": 7, "Ms.": 2, "Dona.": 10}
    for k,v in title_mapping.items():
        titles[titles == k] = v
    DataFrame["Title"] = titles
    return DataFrame


def Creating_NewCols(DataFrame):
    
    DataFrame= Changing_Name_to_Titles(DataFrame)
    DataFrame['NameLen']=DataFrame['Name'].apply(lambda x: len(x))
    DataFrame['FamSize']=DataFrame['SibSp']+DataFrame['Parch']
    cabins=DataFrame['Cabin'].apply(lambda x:   str(x)[0])
    cabin_mapping={'A':3,'B':5,'C':5,'D':4,'E':4,'F':3,'G':2,'T':1,'n':10}
    for k,v in cabin_mapping.items():
        cabins[cabins==k]=v
    DataFrame['Cabin']=cabins
    del DataFrame['Parch']
    del DataFrame['SibSp']
    del DataFrame['PassengerId']
    return DataFrame


def Getting_Dummies(DataFrame,train):
    
    DataFrame=Creating_NewCols(DataFrame)
    pclass = pd.get_dummies( DataFrame.Pclass , prefix='Pclass' )
    sex = pd.get_dummies(DataFrame.Sex)
    embarked = pd.get_dummies(DataFrame.Embarked, prefix='Embarked')
    DataFrame=pd.concat([DataFrame,pclass,sex,embarked],axis=1)
    del DataFrame['Pclass']
    del DataFrame['Name']
    del DataFrame['Sex']
    del DataFrame['Ticket']
    del DataFrame['Embarked']
    DataFrame['Fare'].fillna(train['Fare'].mean(), inplace = True)
    DataFrame['Age'].fillna(train['Age'].mean(), inplace = True)
    return DataFrame


def Create_Decision_Tree(train='train.csv',test='test.csv'):
    trainDF=pd.read_csv(train)
    train=Getting_Dummies(train,trainDF)
    test=Getting_Dummies(test,trainDF)
    DT = DTC(random_state=99)
    DT.fit(train.iloc[:, 1:], train.iloc[:, 0])
    return DT


def Produce_Predictions(FileName,train='train.csv',test='test.csv'):
    trainDF=pd.read_csv(train)
    train=Getting_Dummies(train,trainDF)
    test=Getting_Dummies(test,trainDF)
    DT=Create_Decision_Tree()
    predictions = DT.predict(test)
    predictions = pd.DataFrame(predictions, columns=['Survived'])
    test = pd.read_csv( 'test.csv')
    predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)
    predictions.to_csv(FileName, sep=",", index = False)
    #~ 75% :)
    
"""    
if __name__=="__main__":
    Produce_Predictions('TestRun.csv')
"""