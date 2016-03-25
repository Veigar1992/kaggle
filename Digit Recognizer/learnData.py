from loaddata import loadfile
from loaddata import constructTrainData
from loaddata import writecsv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

def digitRecognizer(data, target, folds):
    for i in range(1, folds+1):
        x_train, x_test, y_train, y_test = train_test_split(data, target,random_state=i)
        knn = KNeighborsClassifier(n_neighbors=10)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        print accuracy_score(y_test, y_pred)

def KNN(X,y,testtrain):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X,y)
    res = []
    for item in testdata:
        tmp = []
        tmp.append(knn.predict(item))
        res.append(tmp)
    return res

if __name__ == '__main__':
    traindata = loadfile('train.csv')
    testdata = loadfile('test.csv')
    X,y = constructTrainData(traindata)
    res = KNN(X,y,testdata)
    writecsv(res, 'res.csv')
