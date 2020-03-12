import re


OPENMP_TEMPLATE = """
#pragma omp parallel
{
  #pragma omp for reduction(+:mean)
  for (int i=0; i<aSize; i++) {
     mean = mean + {};
  }
}
"""


def random_forest(clf, feature_names):
    print('float (*fcnPtr)({});'.format(', '.join(['float'] * len(feature_names))));

def tree2fun(tree, feature_names, fun_name='tree_fun'):
    '''Create tree function header + code
    '''
    print('void {}('.format(fun_name), end='')
    for feature in feature_names:
        print('float {}, '.format(feature), end='')
    print('float *confidences', end='')
    print(') {')
    get_code(tree, feature_names)
    print('}')


def get_code(tree, feature_names):
    '''Create C code from decision tree.

    Based on:
    https://stackoverflow.com/questions/20224526/
    how-to-extract-the-decision-rules-from-scikit-learn-decision-treefter
    '''
    def print_values(val):
        for i, v in enumerate(val[0]):
            print('confidences[{}] = {};'.format(i, v))
        print('return;')
    left = tree.tree_.children_left
    right = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node):
        if threshold[node] != -2:
            print(
                "if ( " +
                features[node] +
                " <= " +
                str(threshold[node])
                + " ) {"
            )
            if left[node] != -1:
                recurse(left, right, threshold, features, left[node])
            print("} else {")
            if right[node] != -1:
                recurse(left, right, threshold, features, right[node])
            print("}")
        else:
            print_values(value[node])

    recurse(left, right, threshold, features, 0)


def slugify(feature_name):
    '''get usable names from feature names, i.e. no spaces,
    no non-alphanumeric characters.
    Example:
    >> list(data.target_names)
    ['setosa', 'versicolor', 'virginica']
    >> feature_names = [
        slugify(feature_name) for feature_name in data.feature_names
    ]
    ['sepal_length__cm_',
     'sepal_width__cm_',
     'petal_length__cm_',
     'petal_width__cm_']
    '''
    if feature_name[0] in [str(n) for n in range(10)]:
        prefix = '_'
    else:
        prefix = ''
    return prefix + re.sub(r'[^a-zA-Z0-9]', '_', feature_name)
