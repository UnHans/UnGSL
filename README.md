# UnGSL
The official codes and implementations of the UnGSL method in the paper: **"Uncertainty-aware Graph Structure Learning"**. 

:+1::+1::+1:**The code is built upon the [OpenGSL benchmark](https://github.com/OpenGSL/OpenGSL). We sincerely appreciate their contributions.** :+1::+1::+1:

## Usage
 One can directly replace the corresponding code in OpenGSL library to use UnGSL. For example, To use GRCN+UnGSL, you can replace `opengsl/module/model/grcn.py` with `GRCNUnGSL/grcn.py` and `opengsl/module/solver/GRCNSolver.py` with `GRCNUnGSL/GRCNSolver.py`.
## Implementation Details in the Experiment
- **Detail for repeated experiments with different random seeds.** Running the base GSL model ten times with different random seeds to obtain node entropy is time-consuming. In our experiments, we first run it three times with different seeds and average the resulting node entropies. Then, we use the averaged entropy vector for multiple trainings of UnGSL.
- **Detail for SUBLIME+UnGSL.** UnGSL uses contrastive learning loss as uncertainty for the unsupervised GSL model SUBLIME. However, CL loss is relatively large, leading to transformed confidence values that are too small. In our experiments, we multiplied the corresponding node confidence vector from SUBLIME by a constant to ensure that the node confidence values are uniformly distributed between 0 and 1.
- **Detail for IDGL+UnGSL on Large-scale datasets.** The IDGL model randomly samples *d* anchor nodes and constructs an *n×d* adjacency matrix (where *d≪n*) during training on large datasets. However, the learnable node-wise thresholds in this scenario cannot be trained via gradient descent because the neighbors of the nodes vary in each epoch. To apply IDGL+UnGSL to large-scale datasets, we randomly sample *d* anchor nodes only in the first epoch and consistently use them in subsequent epochs.
