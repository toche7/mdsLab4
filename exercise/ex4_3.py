def homework(): 
    import pandas as pd
    import numpy as np  
    import matplotlib.pyplot as plt

    from sklearn.datasets import load_iris
    iris = load_iris()
    X = pd.DataFrame(iris.data)
    y = iris.target

    from sklearn.tree import DecisionTreeClassifier
    dt3 = DecisionTreeClassifier(random_state=0)

    dt3.fit(X,y)
    scoreModel = dt3.score(X, y)

    return scoreModel


if __name__ == '__main__':
    print(homework())
