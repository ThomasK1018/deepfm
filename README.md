# DeepFM Model

**DeepFM** (Deep Factorization Machine) is a deep learning model designed for large-scale sparse data in recommendation systems, particularly for tasks like click-through rate (CTR) prediction. It integrates **Factorization Machines (FM)** and **Deep Neural Networks (DNN)** to model both low-order and high-order feature interactions effectively.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)


## Overview

DeepFM combines the strengths of Factorization Machines and Deep Neural Networks. It's especially well-suited for recommendation tasks such as personalized recommendations, CTR prediction, or ranking.

- **FM Component**: Models pairwise feature interactions (low-order interactions) efficiently.
- **DNN Component**: Models high-order feature interactions through deep neural networks.

Both components are trained simultaneously on the same data, leveraging the strengths of each to capture complex relationships in the data.

## Key Features

- **Hybrid Architecture**: Combines FM for low-order and DNN for high-order feature interactions.
- **Efficient for Sparse Data**: Works well with large-scale datasets with sparse categorical features.
- **End-to-End Model**: The model is trained end-to-end, removing the need for manual feature engineering.
- **Flexible**: Suitable for various tasks like recommendation, ranking, and CTR prediction.

## Model Architecture

DeepFM consists of two key parts:
1. **FM Component**: Factorization Machines that handle pairwise feature interactions.
2. **DNN Component**: A deep neural network that captures high-order feature interactions.

The output of both FM and DNN components is combined to produce the final prediction.

## Installation

Clone the repository:
    ```
    git clone https://github.com/ThomasK1018/deepfm.git
    cd deepfm
    ```

## Contributing
If you use this model or code in your research or work, please cite the following paper:
```
@article{author2020deepfm,
  title={DeepFM: A Factorization-Machine based Neural Network for CTR Prediction},
  author={Author, Firstname and Author, Secondname and Author, Thirdname},
  journal={Journal of Machine Learning},
  volume={123},
  number={4},
  pages={567--589},
  year={2020},
  publisher={Publisher Name}
}

```


## Usage

Use it like normal sklearn classifier. The idea of this project is to provide a convenient classification with DeepFM.
