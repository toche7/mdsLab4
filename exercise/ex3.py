def homework(): 
    import pandas as pd
    import numpy as np  
    import matplotlib.pyplot as plt

    exDf = pd.read_csv("https://raw.githubusercontent.com/toche7/DataSets/main/yieldvsTemp.txt")
    exDf

    # create model 1 for linear regression

    from sklearn.linear_model import LinearRegression
    linreg = LinearRegression()
    X = exDf[['Temp']]
    y = exDf['Yield']
    linreg.fit(X, y)
    yhat = linreg.predict(X)
    scoreModel1 = linreg.score(X, y)
    
    # plt.plot(X, y, 'o')
    # plt.plot(X, yhat, '-')
    # plt.show()


    # create model 2 for linear regression wih polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    linreg.fit(X_poly, y)
    yhat = linreg.predict(X_poly)
    scoreModel2 = linreg.score(X_poly, y)
    #print(scoreModel2)
    # plt.plot(X, y, 'o')
    # plt.plot(X, yhat, '-')
    # plt.show()

    return scoreModel1, scoreModel2


if __name__ == '__main__':
    print(homework())
