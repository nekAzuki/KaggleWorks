# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.optimize import fmin_bfgs, fmin #optimization function

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# theta ?x1
# X format [dummy Pclass Sex Age SibSp Parch Fare]

def signomid(z):
    result = np.zeros(z.shape)
    for i in range(0,z.shape[0]):
        for j in range(0,z.shape[1]):
            result[i][j] = 1 / (1 + np.exp(-z[i][j]))
    return result


def costfunction(theta, X, y, lambd):
    theta = np.asmatrix(theta).T
    y = np.asmatrix(y).T
    cost = 0
    #gradient = np.zeros((theta.shape[0],1))
    m = X.shape[0]
    for i in range(0, m):
        hypo = signomid(X[i] * theta)[0][0]
        cost += -y[i][0] * np.log(hypo) - (1-y[i][0]) * np.log(1 - hypo)
    cost /= m
    
    for i in range(0, theta.shape[0]):
        cost += lambd * pow(theta[i][0], 2) / m
    
    
    # for i in range(0, m):
    #     gradient[0][0] += (signomid(X[i] * theta) - y[i][0]) * X[i][0]
    # gradient[0][0] /= m
    
    # for j in range(1, gradient.shape[0]):
    #     for i in range(0,m):
    #         gradient[j][0] += (signomid(X[i] * theta) - y[i][0])
    #     gradient[j][0] /= m;
    #     gradient[j][0] += lambd / m * theta[j][0]

    #return [cost, gradient]
    #print(cost)
    return cost

def gradient(theta, X, y, lambd):
    theta = np.asmatrix(theta).T
    y = np.asmatrix(y).T
    
    gradient = np.zeros((theta.shape[0],1))
    m = X.shape[0]
    for i in range(0, m):
        gradient[0][0] += (signomid(X[i] * theta) - y[i][0]) * X[i][0]
    gradient[0][0] /= m
    
    for j in range(1, gradient.shape[0]):
        for i in range(0,m):
            gradient[j][0] += (signomid(X[i] * theta) - y[i][0])
        gradient[j][0] /= m;
        gradient[j][0] += lambd / m * theta[j][0]
    
    return np.squeeze(np.asarray(gradient))
 
        
def load_data():
    csv = pd.read_csv('../input/train.csv')
    y = csv['Survived']
    y = y.as_matrix()
    csv = csv.drop(['PassengerId','Survived', 'Name', 'Cabin', 'Embarked', 'Ticket'], axis=1)
    csv['Age'].fillna(0, inplace=True)
    csv['Age'].replace(0, csv['Age'].sum() / len(csv), inplace=True)
    csv['Sex'].replace('female', 0, inplace=True)
    csv['Sex'].replace('male', 1, inplace=True)
    csv = csv.as_matrix()
    csv = np.insert(csv, 0, 1, axis=1)
    return [csv, y]
    
[X, y] = load_data()
theta = np.zeros((X.shape[1],1))
theta_opt = fmin(costfunction, theta, args=(X, y, 1), xtol=0.0005)

def solve(theta):
    csv = pd.read_csv('../input/test.csv')
    ids = csv['PassengerId']
    ids = np.squeeze(np.asarray(ids.as_matrix()))
    csv = csv.drop(['PassengerId', 'Name', 'Cabin', 'Embarked', 'Ticket'], axis=1)
    csv['Age'].fillna(0, inplace=True)
    csv['Age'].replace(0, csv['Age'].sum() / len(csv), inplace=True)
    csv['Sex'].replace('female', 0, inplace=True)
    csv['Sex'].replace('male', 1, inplace=True)
    csv = csv.as_matrix()
    csv = np.insert(csv, 0, 1, axis=1)
    
    
    m = csv.shape[0]
    
    ans = []
    for i in range(0, m):
        print(csv[i])
        print(theta)
        if signomid(csv[i] * np.asmatrix(theta).T) >= 0.5:
            ans.append(1)
        else:
            ans.append(0)

    df = pd.DataFrame({'PassengerId':pd.Series(ids), 'Survived':pd.Series(ans)})
    df.to_csv('ans.csv', index=False)
    
solve(theta_opt)

        
        
        
        
        
        
        
        
        
        
        