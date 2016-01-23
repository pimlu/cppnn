## C++ neural networks

This is a multilayer perceptron implementation written in C++ with Eigen.  Everything is vectorized with Eigen - there aren't any nested loops used to evaluate/train the network.

It supports any number of fully connected layers of any size.  It trains using backpropagation.  It supports RPROP as a weight update mechanism.

All of the important code is in `src/MLP.cpp`.

### Building

Provided you have a C++ compiler and Eigen headers installed, you should be able to just run `make`.  Depending on your installation you make have to refer to your Eigen headers with `-I`, which can be specified using `OPT` in the makefile.

### Running

The binary will be in `dist/cppnn`.  Right now `main.cpp` includes a basic example that does training of a small (not minimal) network to fit XOR.  `mnist.cpp` (which `main` calls right now) trains on MNIST data.  As it is, it gets about 85% accuracy after a minute and 20 seconds of training.  It needs the MNIST data in a directory you pass in as an argument.

### //TODO
 * RPROP tweaks (need to limit weights, among other things)
 * Parallelization
 * MNIST preprocessing (width/brightness normalization, experiment with DCT)