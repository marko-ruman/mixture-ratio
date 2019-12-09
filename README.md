# Mixture Ratio

Implementation of a novel Mixture Ratio probabilistic model. 

Mathematical details can be found in the publications folder

# Usage


```
from mixture_ratio import MixtureRatio

#setting domain of all variables in (y, X)
variables_domain = [10, 5, 7, 3, 2]

#setting the mixture's structure
variables_connection = [[0, 1, 2], [0, 3], [0, 4]]

#initializing the MixtureRatio object
mixture = MixtureRatio(variables_domain, variables_connection)

#fitting the data
mixture.fit(X, y)

# predicting probability of all classes for the given X
mixture.predict_proba(X)
```

