import sys

def gini(X, TOL=sys.maxsize):
    total_samples = sum(X)  # Total number of samples
    gini = 1.0
    if total_samples==0:
        return 0

    if len(X) < TOL:
        # Calculate the Gini impurity
        for count in X:
            proportion = count / total_samples
            gini -= proportion ** 2
    else:
        raise ValueError(f"List size exceeds the maximum tolerance of {TOL}.")
    return gini

X = [1,2,3,4]
res = gini(X, TOL=3)
print(res)