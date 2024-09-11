
## Project Structure
```
recsys-project/
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
│   ├── training/                     # Training scripts and pipeline
│   │   ├── train_baseline.py         # Training script for baseline models
│   │   └── ...
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
│   ├── matrix_factorization/
│   └── ...
│
├── tests/                            # Unit and integration tests
│   ├── test_data_loader.py
│   ├── test_metrics.py
│   └── test_model_training.py
│
├── requirements.txt                  # Required Python packages
├── README.md                         # Project overview, instructions, and documentation
├── .gitignore                        # Ignore unnecessary files (e.g., data, checkpoints, etc.)
└── setup.py                          # Setup file for packaging and dependencies

```

Breakdown of Directories and Files:
data/:

Contains raw datasets, such as MovieLens, and preprocessed versions.
You can add datasets from other sources or versions of MovieLens (e.g., 1M, 10M) here.
notebooks/:

Jupyter/Colab notebooks for initial experimentation, exploratory data analysis (EDA), and model comparison.
Useful for running quick tests and visualizations.
src/:

data_loader.py: Responsible for loading raw datasets, applying preprocessing, and transforming data into formats suitable for model training.
movielens_utils.py: Specific utilities for handling the MovieLens dataset (e.g., downloading, splitting data).
metrics.py: Contains evaluation metrics, like RMSE, MAE, Precision@K, Recall@K, and NDCG.
models/: Includes all the recommendation algorithms (e.g., Matrix Factorization, Neural Collaborative Filtering, Transformers).
training/: Contains scripts for training the models, managing experiments, and pipelines.
utils/: General utility scripts such as logging, model checkpointing, and early stopping.
experiments/:

Stores experiment configuration files (e.g., YAML or JSON) that define the settings (e.g., hyperparameters, dataset splits) for each model.
Allows for quick switching between different configurations to run experiments efficiently.
results/:

Holds results, including logs, evaluation metrics, and saved model checkpoints for each experiment.
Organized by model type for easy reference.
tests/:

Unit tests for key parts of the codebase (e.g., data loading, model training, evaluation).
Keeps your codebase robust and ensures that changes don’t break functionality.
Other Files:

requirements.txt: Lists all Python dependencies needed for the project, e.g., pandas, torch, scikit-learn, etc.
README.md: Provides an overview of the project, including setup instructions, dataset details, and how to run models.
.gitignore: Ensures that large files, unnecessary logs, and datasets are not included in the repository.
setup.py: Useful if you plan to package this project or share it more formally.
Example Flow for Using the Repo:
Load the Dataset:

The data_loader.py or movielens_utils.py will handle the downloading and preprocessing of the dataset.
Experimentation in Notebooks:

Use the notebooks/ directory for performing EDA or testing different models interactively.
Model Training:

Train models using the scripts in src/training/ and save results and checkpoints in the results/ directory.
Running Experiments:

Use configurations in the experiments/ directory to run and compare different models and hyperparameter settings.
Additional Considerations:
Modular Structure: The modular design allows for easy integration of new recommendation models, datasets, and utilities without cluttering the project.
Experiments & Hyperparameters: Storing experiment configurations separately makes it easier to reproduce experiments, tweak hyperparameters, and run batch experiments.
Testing: Include unit and integration tests to ensure your code remains functional and scalable as the project grows.
This structure ensures that your project remains well-organized, easy to scale, and maintains reproducibility across different experiments.

## Repo Reference:

* https://github.com/nntrongnghia/learn-recsys/tree/main