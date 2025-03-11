# Drop Activation Comparison
This repository is created to show the comparison of Drop Activation in [this paper](https://arxiv.org/abs/1811.05850) with 2 datasets - [Street View House Numbers (SVHN)](http://ufldl.stanford.edu/housenumbers/) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).
We chose 2 model architectures - [LeNet](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) and [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), and trained them with various modifications to prevent overfitting.
The modifications are:
1. [Dropout](http://arxiv.org/abs/1207.0580)
2. [Batch Normalization](http://arxiv.org/abs/1502.03167)
3. [L1 and L2 Regularization](https://doi.org/10.1145/1015330.1015435)
4. [Drop Activation](https://arxiv.org/abs/1811.05850)

The Drop Activation is sourced from the [GitHub repository](https://github.com/LeungSamWai/Drop-Activation/blob/master/models/cifar/da_wrn.py) mentioned in its paper. It was ported from PyTorch to TensorFlow.
The Drop Activation notebooks contain a combination of Dropout and BatchNormalization along with Drop Activation itself to show that Drop Activation is compatible with other overfitting prevention techniques.
The models are also trained without any modifications in order to provide a baseline for comparison of results.

# Order of Notebooks
Two datasets, each with two model architectures and each architecture with 5 variations - baseline (no modifications),  Dropout, BatchNormalization, L1L2Regularization and DropActivation (Dropout + BatchNormalization + Drop Activation), leading to 20 Jupyter notebooks in total.
```
SVHN
├── LeNet
│   ├── 1. LeNet
│   ├── 2. Dropout
│   ├── 3. BatchNormalization
│   ├── 4. L1L2Regularization
│   └── 5. DropActivation
└── AlexNet
    ├── 6. AlexNet
    ├── 7. Dropout
    ├── 8. BatchNormalization
    ├── 9. L1L2Regularization
    └── 10. DropActivation
CIFAR10
├── LeNet
│   ├── 11. LeNet
│   ├── 12. Dropout
│   ├── 13. BatchNormalization
│   ├── 14. L1L2Regularization
│   └── 15. DropActivation
└── AlexNet
    ├── 16. AlexNet
    ├── 17. Dropout
    ├── 18. BatchNormalization
    ├── 19. L1L2Regularization
    └── 20. DropActivation
```

# How To Run The Colab Notebooks
The Colab notebooks are entirely self-contained i.e., dataset download, dataset preprocessing, model creation and model training are entirely within the notebooks and do not require any additional library download or installation.
All code within is written using Keras bundled with TensorFlow. Running all the cells in the notebook from start to end will download the dataset, preprocess it, create the model, train it, plot a graph of training accuracy vs validation accuracy and show the final accuracy on the testing set.
Additionally, the entire model is saved in the runtime with aptly named files after the entire training completes. This should help in validating the accuracy yourself and also deploying it if necessary.

# Thanks
Thanks to [Google](https://www.google.com) for providing free access to [Colab](https://colab.research.google.com/) execution environment.
