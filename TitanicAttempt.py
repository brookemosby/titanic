import pandas as pd
from sklearn.tree import DecisionTreeClassifier as DTC

def Read_Files(FileName):
    """Accepts file name, reads in file, converts and returns pd.DataFrame type"""
    DataFrame=pd.read_csv(FileName)
    return DataFrame


def Changing_Name_to_Titles(DataFrame):
    """Accepts pd.DataFrame and takes from column name and creates column title filled with numbers corresponding to title type"""
    DataFrame= Read_Files(DataFrame)
    titles=DataFrame['Name'].apply(lambda x: x.split(',')[1].split(' ')[1])
    title_mapping = {"the":7, "Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Dr.": 5, "Rev.": 6, "Major.": 7, "Col.": 7, "Mlle.": 8, "Mme.": 8, "Don.": 9, "Lady.": 10, "Countess.": 10, "Jonkheer.": 10, "Sir.": 9, "Capt.": 7, "Ms.": 2, "Dona.": 10}
    for k,v in title_mapping.items():
        titles[titles == k] = v
    DataFrame["Title"] = titles
    return DataFrame


def Creating_NewCols(DataFrame):
    """Accepts pd.DataFrame and creates columns NameLen- len of name string & FamSize- # of siblings addded to # of parents & children.
    Modifies Cabin column so that the cabin letter is mapped to a number corresponding to the cabin.
    Deletes columns 'Parch', 'SibSp', and 'PassengerId', and then returns DataFrame
    """
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
    """Accepts DataFrame, and training set DataFrame, Creates dummmies for columns 'Pclass', 'Embarked', and 'Sex'.
    Then deletes previous columns, along with 'Ticket', and fills in missing values for 'Age' and 'Fare'.
    Returns DataFrame
    """
    
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


def Create_Decision_Tree(train='train.csv'):
    """Creates Decision Tree fitted with default train='train.csv', returns Decision Tree
    """
    trainDF=pd.read_csv(train)
    train=Getting_Dummies(train,trainDF)
    DT = DTC(random_state=99)
    DT.fit(train.iloc[:, 1:], train.iloc[:, 0])
    return DT


def Produce_Predictions(FileName,train='train.csv',test='test.csv'):
    """Accepts file name, 'example.csv', to print predictions too, along with default train='train.csv' & test='test.csv'.
    Uses Decision Tree to create predictions on who survived.
    returns nothing, creates 'example.csv'
    """
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