# **LDMGI: Local Discriminant Models and Global Integration**

**Image Clustering Using Local Discriminant Models and Global Integration**
*IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2010*
🔗 [Paper Link (IEEE Xplore)](https://ieeexplore.ieee.org/document/5481703)

------

## **Introduction**

This project implements the **LDMGI algorithm** for unsupervised image clustering, combining local discriminant models with global integration for robust performance. Key features:

- Supports datasets: **JAFFE, UMIST, MNIST, MPEG7, USPS, COIL-20**.
- Includes **data preprocessing, parameter tuning (λ), and evaluation** (ACC/NMI).
- Saves clustering results for further analysis.

------

## **Usage**

### **1. Setup**

Ensure MATLAB is installed, and clone this repository:

```bash
git clone https://github.com/your-repo/LDMGI.git
cd LDMGI
```

### **2. Data Preparation**

Preprocess datasets into `(X, Y)` format using provided scripts:

- `loadJAFFE.m`: Loads JAFFE facial expression images and labels.
- `loadUSPS.m`: Loads USPS handwritten digits.
- Other datasets: See `./functions/`.

Example for JAFFE:

```matlab
load('JAFFE.mat');
X_comb =  X_JAFFE;
Y_comb = Y_JAFFE;
Y_comb = Y_comb-1; % Every label should be transfer into 0~(c-1) for further data processing
```

### **3. Running Clustering**

#### **Single Dataset Example (JAFFE)**

Edit and run `main_JAFFE.m`:

```matlab
% Parameters
c = 10;                 % Corresponds to the number of cluster of your dataset
k = 5;                 % Neighborhood size
lambda_list = [1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6, 1e8];  

% Run clustering
for lambda = lambda_list
    Y = ldmgi_clustering(X, c, 'k', k, 'lambda', lambda);
    [acc, nmi] = evaluate_clustering(Y, Y_true);
    fprintf('λ=%.0e: ACC=%.4f, NMI=%.4f\n', lambda, acc, nmi);
    save(sprintf('Y_JAFFE_lambda_%.0e.mat', lambda), 'Y');
end
```

Therefore, by applying the shape of this **X : (m, n)**, where m denotes the feature space and n denotes the number of samples, **Y = (1, n)**. You can use LDMGI for any dataset.

However, in our test, we have found that the computation time is highly related to the size of Lagrangian: (n, n). We recommend the largest data size is ~7000.    

### **4. Outputs**

- **Results**: Saved in `./Experiment_data/DATASET_NAME/` (e.g., `Y_JAFFE_lambda_1e4.mat`).

- **Accuracy and NMI**: After running a certain lambda, the output is like this (Taking MNIST-T as example): 
  Running experiment with lambda = 1e-08...
  Computed Global Laplacian Matrix L:
          5000        5000

  Accuracy: 74.00%
  Lambda: 1e-08, ACC: 0.7400, NMI: 0.7268

------

## **Key Functions**

| Function                  | Description                                       |
| :------------------------ | :------------------------------------------------ |
| `ldmgi_clustering.m`      | Core LDMGI algorithm implementation.              |
| `calculate_cost_matrix.m` | Computes assignment cost for Hungarian algorithm. |
| `evaluate_clustering.m`   | Calculates ACC and NMI metrics.                   |

------

## **Example Results**

| λ (Lambda) | ACC (JAFFE) | NMI (JAFFE) |
| :--------- | :---------- | :---------- |
| 1e-8       | 0.72        | 0.65        |
| 1e4        | 0.85        | 0.78        |

The data of each generated Y corresponds to certain lambda is stored in the direction of experiment data. 

------

## **Dependencies**

- MATLAB ≥ R2020b
- Statistics and Machine Learning Toolbox

------

### **Need Help?**

For issues, please open a **GitHub Issue** or contact:
📧 Email: [u3637086@connect.hku.hk](https://mailto:your-email@example.com/)

------

