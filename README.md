# Compile sklearn models to C/C++

```python
# import some dataset:
from sklearn.datasets import load_iris
data = load_iris()

# create a tree model:
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data.data, data.target)

# create a c function from the tree:
from sklearn_compile.toC import *
feature_names = [slugify(feature_name) for feature_name in data.feature_names]
tree2fun(tree=clf, feature_names=feature_names, fun_name='tree_0')
```