# What is nn-from-scratch?
nn-from-scratch is a custom Python library where neural networks are written from scratch. 

# Why reinvent the wheel?
Since 2010, machine learning (ML) and, by extension, deep learning (DL) have exploded in popularity. A large number of open source ML/DL libraries have since been made available, such as Tensorflow, Theano, and PyTorch. These libraries have made deep learning highly accessible, where models can be designed and implemented with a few lines of code.

The increased accessibility came at a cost of understanding the underlying science. Projects that use these libraries makes an assumption of "just trust the code". Thus, the machinery that works behind the scenes are taken for granted. This does not sit well with the mathematician in me because it is in our nature to question why things work, and under what conditions do things break? This repository exists to address these questions, by tearing down all the libraries that have been taken for granted then rebuilding them from the ground up starting with the mathematics.

# What this isn't
This library is meant to gain a better understanding of deep learning from scratch. Speed and memory efficiencies will be considered but will not be a priority. The code will be written using NumPy, and perhaps be extended to CuPy for GPU computation when the code becomes sufficiently complex.
