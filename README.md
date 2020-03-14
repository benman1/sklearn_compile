# Compile sklearn models to C/C++

## how to use (currently):
```python
# import some dataset:
from sklearn.datasets import load_iris
data = load_iris()

# create a c function from the tree:
from sklearn_compile.toC import *
feature_names = [
    slugify(feature_name)
    for feature_name in data.feature_names
]
source_string = tree2fun(
    tree=clf,
    feature_names=feature_names,
    fun_name='tree_0'
)
header = """
void tree_0(float sepal_length__cm_, float sepal_width__cm_, float petal_length__cm_, float petal_width__cm_, float *confidences);
"""

# compile the tree:
import cffi
ffi = cffi.FFI()

ffi.cdef(header)
ffi.set_source(
    'mytree',
    source_string,
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)
ffi.compile()

# use the tree:
import mytree
import numpy as np

confidences = np.zeros(3, dtype=np.float32)
_confidences = mytree.ffi.cast('{} *'.format('float'), mytree.ffi.from_buffer(confidences))

# finally:
mytree.lib.tree_0(0.0, 5.0, 10.0, 10.0, _confidences)
```

## status
This works for the individual trees, I haven't, however, put it together, so there's no aggregation of trees into forests at the moment. The idea would be as follows:
# organise the project properly using [a cookiecutter](https://github.com/audreyr/cookiecutter-pypackage)
# integrate trees into forests using [a template with both OpenMP and OpenACC](https://github.com/benman1/parallel_reduction_example/blob/master/reduction.cpp)
# template trees and forest using [mako](https://docs.makotemplates.org/en/latest/usage.html#basic-usage)

This would already be nice; both OpenMP and OpenACC support multicore CPU and GPU out-of-the-box. FPGA could work with smaller tweaks.