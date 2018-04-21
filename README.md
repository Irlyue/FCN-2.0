- configs/: for configurations
- inputs/: for data input API
- models/: for model zoo
- notebooks/: for examination
- pys/: for one-time python scripts
- scripts/: for bash scripts
- utils/: for utility functions, basically there are two main kinds
  - os utility(or file utility)
  - tf utility(for TensorFlow utility)
- my_utils.py: may just include the following lines
```Python
from utils.tf_utils import *
from utils.os_utils import *
```