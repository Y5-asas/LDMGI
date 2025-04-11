# LDMGI: Local Discriminant Models and Global Integration

> **Image Clustering Using Local Discriminant Models and Global Integration**  
> *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2010*  
> [🔗 Paper Link (IEEE Xplore)](https://ieeexplore.ieee.org/abstract/document/5454426)

---

## Introduction

This project implements the LDMGI algorithm for unsupervised image clustering.  
It includes feature preprocessing, parameter sweeping, and clustering evaluation across multiple datasets such as JAFFE, UMIST, MNIST, and MPEG7.

Before running the algorithm, each dataset must be preprocessed into matrix format (`X`, `Y`) using the provided preprocessing scripts.

---

## Usage

The main clustering process is demonstrated in `main_JAFFE.m`, which serves as a template for other datasets as well.

```matlab
% Example:
main_JAFFE.m
```
## Dependencies
MATLAB ≥ R2020b

Statistics and Machine Learning Toolbox
