# Natural Language Processing Project

This repository contains a Natural Language Processing (NLP) project aimed at exploring and modeling text data. The project is structured to facilitate data loading, preprocessing, feature extraction, model training, and evaluation.

## Project Structure

- **data/**: Contains raw and processed data files.
  - **raw/**: Directory for raw data files.
  - **processed/**: Directory for cleaned and processed data files.
  
- **notebooks/**: Jupyter notebooks for exploratory data analysis.
  - **01-exploration.ipynb**: Notebook for visualizing and understanding the data.

- **src/**: Source code for the project.
  - **data/**: Data loading functions.
    - **loader.py**: Functions to load raw and processed data.
  - **preprocessing/**: Text cleaning functions.
    - **text_cleaning.py**: Functions for cleaning text data.
  - **features/**: Feature extraction methods.
    - **vectorizer.py**: Classes/functions for converting text to numerical features.
  - **models/**: Model architecture definitions.
    - **model.py**: Classes for different model types.
  - **training/**: Model training logic.
    - **train.py**: Functions to train and save the model.
  - **evaluation/**: Model evaluation functions.
    - **evaluate.py**: Functions to assess model performance.

- **experiments/**: Documentation for experiments conducted.
  - **README.md**: Details of different model configurations and results.

- **configs/**: Configuration files.
  - **config.yaml**: Parameters for file paths, model hyperparameters, and preprocessing options.

- **requirements.txt**: List of Python dependencies required for the project.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```
   cd nlp-project
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

- Use the Jupyter notebook in the `notebooks/` directory for exploratory data analysis.
- Modify and run scripts in the `src/` directory for data processing, model training, and evaluation.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.