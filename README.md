# H&M Personalized Fashion Recommendations

This repository contains a personalized fashion recommendation system for H&M customers. Main aim of the project is to create a visually seperated and personalized recommendation system with different emthods and compare them. This work has been done for the BLG 607 Data Mining course which is given by the Prof. DR Şule Gündüz Öğüdücü in Istanbul Technical University.

## Requirements

To run this project, you will need the following:

- Python 3.7 or higher
- Jupyter Notebook
- Required Python libraries 
    - implicit
    - streamlit
    - torch
    - wandb
    - other general libraries (pandas, numpy, scikit-learn)


## Data

The dataset used for this project is H&M Personalized Fashion Recommendations which includes customer information, product details, and transaction history. Dataset can be downloaded from https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data  . The data is stored in the `data` directory and includes the following files:

- `customers.csv`: metadata for each customer_id in dataset
- `articles.csv`: detailed metadata for each article_id available for purchase
- `transactions.csv`: he training data, consisting of the purchases each customer for each date, as well as additional information. Duplicate rows correspond to multiple purchases of the same item. Your task is to predict the article_ids each customer will purchase during the 7-day period immediately after the training data period.
- `images\`: A folder of images corresponding to each article_id; images are placed in subfolders starting with the first three digits of the article_id; note, not all article_id values have a corresponding image.

## Model Weights

The pre-trained model weights msut be stored in the `weights` directory.  You can load these weights to make predictions without retraining the model. Weights can be downloaded from https://drive.google.com/drive/folders/1tY_NE-y1ReUgyW8NgUi_R-D7NKTRtal2?usp=sharing

## Usage

To run the recommendation system, follow these steps:

1. Clone the repository:
    ```bash
    https://github.com/HFDurkaya/Data_mining_recsys
    cd Data_mining_recsys
    ```

2. Install the required libraries:
    ```bash
    pip install implicit torch streamlit pandas etc.
    ```
3. Run the notebook `data_analyaia.ipynb` to make exploratory data analysis on  dataset. This notebook includes steps for understanding the dataset in a deeper manner.

4. Run the notebook `dataset_optimization.ipynb` to preprocess dataset. This notebook includes steps for data preprocessing, csv files will be converted to pickle files after this notebook.

5. Run the notebook `model_testing.ipynb` to generate fashion recommendations. This notebook includes steps for data preprocessing, model training, and generating recommendations.

6. Run the notebook `demo.py` to generate fashion recommendations in a user interface from streamlit.
    ```bash
    streamlit run demo.py
    ```

 


## Evaluation (evaluate.py)

The evaluation pipeline is designed to streamline the training and evaluation of recommendation models. It supports two modes of operation:

- **Single Evaluation Mode**: Runs a single experiment using a fixed configuration.
- **Grid Search Sweep Mode**: Performs hyperparameter tuning using Weights & Biases (W&B) sweeps.

### How the Pipeline Works

**Input Arguments**:
- `--config`: Path to the YAML configuration file containing experiment settings (required).
- `--results-dir`: Directory to save evaluation results (default: Results).

**Configuration Loading**: The pipeline reads the YAML configuration file to extract:
- Data paths
- Model parameters
- Run mode (single or sweep)
- W&B project name

**Pipeline Initialization**: The following components are initialized:
- **DataLoader**: Loads datasets required for model evaluation.
- **MetricCalculator**: Computes evaluation metrics (e.g., precision, recall).
- **RecommenderEvaluator**: Evaluates the models using the provided metrics.
- **ModelFactory**: Builds recommendation models dynamically.
- **ResultsManager**: Handles saving evaluation outputs.
- **ExperimentManager**: Coordinates the entire experiment workflow.

### Modes of Operation

- **Single Evaluation Mode**:
    - Runs one evaluation experiment with the provided configuration.
    - Logs experiment details and metrics to W&B for easy tracking.

- **Grid Search Sweep Mode**:
    - Uses W&B sweeps to perform grid search hyperparameter tuning.
    - Combines fixed parameters from the configuration file with sweep-specific parameters to optimize the model.

**Error Handling**: The pipeline captures and logs errors during execution to help with debugging.

### Example Usage

#### Run a evaluation experiment
```bash
python main.py --config path/to/config.yaml --results-dir Results
```


### W&B Integration

This pipeline is integrated with Weights & Biases (W&B) for experiment tracking and hyperparameter optimization. Key features include:

- Logging experiment results and metrics in real-time.
- Automating hyperparameter tuning using W&B sweeps.
- Visualizing performance across multiple runs.



## Additional Information (Folders)


- **wandb**: Stores the experiment results form wandb.
- **Weights**: Pre-trained model weights.
- **Utils**: Utility functions for visualizing and demonstrating recommendations.
- **Results**: Contains experiment results.
- **recommendations**: Stores recommendation results.
- **Models**: Stores model files.
- **eda_result_images**: Stores EDA plots.
- **data**: Stores dataset csvs, pickle files and images.
- **configs**: Stores configs for the wandb experiments.
