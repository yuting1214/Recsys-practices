
## Quick-Start

* [Colab Link](https://colab.research.google.com/drive/1-3Y9nl0s1eNTUOslXquTt9okSzCLMytx?usp=sharing)

## Project Structure

<details>
<summary>More Project Details</summary>
```
recsys-practices/
│
├── data/                             # Directory for datasets and preprocessed data
│   ├── movielens/                    # MovieLens datasets
│   │   ├── ml-100k/
|   │   │   ├── ml-100k.zip               # Raw dataset files (downloaded)
|   │   │   ├── u.data                    # Extracted dataset
|   │   │   ├── processed_data.csv         # Processed dataset (optional)
│   └── other_datasets/               # Placeholder for other datasets if needed
│
├── notebooks/                        # Jupyter/Colab notebooks for experimentation
│   ├── exploratory_data_analysis.ipynb
│   ├── baseline_model_evaluation.ipynb 
│   └── sota_algorithm_comparison.ipynb
│
├── src/                              # Source code for models, data processing, and utilities
│   ├── __init__.py                   # Makes src a package
│   ├── metrics.py                    # Evaluation metrics for recommendations (e.g., RMSE, Precision@K)
│   ├── data_loader/                     # Training scripts and pipeline
│   │   ├── DataModule.py
│   │   └── data_loader.py
│   ├── models/                       # Directory for model classes and algorithms
│   │   ├── ModelModule.py 
│   │   ├── baseline.py               # Simple models (e.g., User-Mean, Item-Mean)
│   │   ├── matrix_factorization.py   # Matrix Factorization models (e.g., ALS, SVD)
│   │   ├── neural_collaborative.py   # Neural Collaborative Filtering models
│   │   ├── deepfm.py                 # DeepFM model
│   │   ├── transformers.py           # Transformer-based recommendation models
│   │   └── ...                       # Other SOTA algorithms like LightFM, BERT4Rec, etc.
│   └── utils/                        # General utilities (e.g., logging, saving models)
│       ├── logger.py
│       ├── checkpoint.py             # Saving/loading model checkpoints
│       └── early_stopping.py         # Early stopping implementation
│
├── experiments/                      # Directory for running different experiments
│   ├── baseline_config.yaml          # Configuration for baseline model experiments
│   ├── neural_collaborative_config.yaml
│   └── transformer_config.yaml       # Configuration for transformer-based model experiments
│
├── results/                          # Directory for saving model results, logs, and visualizations
│   ├── baseline/
│   │   ├── results.csv
│   │   ├── logs/
│   │   └── checkpoints/
│   └── ...
│
├── requirements.txt                  # Required Python packages
├── README.md                         # Project overview, instructions, and documentation
└──  .gitignore                        # Ignore unnecessary files (e.g., data, checkpoints, etc.)

```
</details>

## Repo Reference:

* https://github.com/nntrongnghia/learn-recsys/tree/main