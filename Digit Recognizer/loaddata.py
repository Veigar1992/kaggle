import csv

def loadfile(filename):
    tmp = csv.reader(file(filename,'rb'))
    csvdata = []
    flag = True

    for line in tmp:
        if flag == True:
            flag = False
            continue
        csvdata.append(line)
    print type(csvdata), csvdata[0]
    return csvdata

def constructTrainData(csvdata):
    X=[];y=[]
    for line in csvdata:
        y.append(line[0])
        del line[0]
        X.append(line)
    return X, y

def writecsv(data,filename):
    tmp = file(filename,'wb')
    wr = csv.writer(tmp)
    wr.writerows(data)
    tmp.close()

if __name__ == '__main__':
    csvdata = loadfile('train.csv')