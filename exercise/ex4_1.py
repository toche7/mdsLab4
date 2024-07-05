def homework(): 
    import pandas as pd
    import numpy as np  
    import matplotlib.pyplot as plt

    import pandas as pd
    data = pd.read_csv("https://raw.githubusercontent.com/toche7/DataSets/main/admit.csv")
    y = data.Label
    X = data[['SubjectA','SubjectB']]
    from sklearn.neural_network import MLPClassifier
    nn = MLPClassifier(solver='lbfgs', alpha=1e-5,
                   hidden_layer_sizes=(10,10),
                   random_state= 10,
                   max_iter = 1000)
    nn.fit(X, y)
    scoreModelNN = nn.score(X, y)
    return scoreModelNN


if __name__ == '__main__':
    print(homework())
