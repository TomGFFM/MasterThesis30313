# Python project related to the Master Thesis: Exploring the Adaptation and Generalization Capabilities of Transformer-Based Reinforcement Learning Agents in Atari Gaming Environments

by Thomas Graf

## Project Description
This research investigates the use of transformer-based neural networks in the field of reinforcement learning (RL) in Atari games and explores their adaptability, performance and generalisation capabilities compared to long short term memory (LSTM) networks.

## Features
- Support for the use of Transformer and LSTM architectures in an ALE RF environment.
- Includes Replay Buffer, Prioritized Replay, and Noisy Layers implementations.
- Analysis of training results and Gameplay of trained models.

## Setup and Installation

### Installation via Conda or Mamba package manager 

- If not available install conda, anaconda or mamba from the relevant resources for your platform e.g.:

  - https://github.com/conda-forge/miniforge
  - https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html
  - https://docs.anaconda.com/anaconda/install/
 
- Execute the environment creation using the prepared **environment.yml** file:
  - ```bash
    conda env create -f environment.yml
    ```
  alternatively use mamba:
  - ```bash
     mamba env create -f environment.yml
      ```

### Installation via basic Python Installation
1. **Install Python** version 3.9.19 from https://www.python.org/downloads/
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Install Atari ROMs**:
   Use [AutoROM](https://github.com/Farama-Foundation/AutoROM) from the Farama Foundation to install the required Atari ROMs by executing:
   ```bash
   AutoROM
   ```
   Follow the instructions provided by AutoROM to complete the installation.

## Project Structure Overview
```
.
├── _old                    # Unused and outdated versions of the software
├── agents                  # Agent class which is used in the RF learning environment
├── analysis                # Jupyter notebooks for analysing training results
├── game_play               # Scripts to execute trained models for active gameplay
├── networks                # Neural network architectures for transformers, ltsms and cnns
├── output                  # Output files from model training
    ├── metrics             # Yaml files from all optuna trials containing collected metrics 
    ├── model_extractions   # Image files which were extracted from the final CNN layer
    ├── models              # Model checkpoints from all Optuna trials
    ├── optuna_study        # Optuna study files which can be loaded in a dashboard
├── training                # Training scripts which define the env, network type and hyperparameters 
├── utils                   # Utility scripts incl. pre-processing, training loop, replay buffer and others
├── requirements.txt        # Python dependencies
├── environment.yml         # Yaml file for creating the full Python environment via Conda
└── LICENSE                 # License
```

## Usage
### Training Scripts
Move to the ```training``` folder and configure ALE environment and hyperparameters as desired. Execute the training for 
each model type by performing the following steps. Be aware that each execution is causing files to be created in the ```output``` folder:
- **Transformer Model**:
  ```bash
  python ddqn_transformer_space_invaders_training_v9OptunaPrioReplayNoisy.py
  ```
- **LSTM Model**:
  ```bash
  python ddqn_lstm_space_invaders_training_v9OptunaPrioReplayNoisy.py
  ```

### Gameplay Scripts
To execute the trained models from the relevant checkpoint you can use the scripts from the ```gameplay``` folder. Be aware 
that within the scrips the correct parameter yaml files and the matching model checkpoints have to be entered manually.
In the current version of the scripts you can see working examples.

- **Transformer Model**:
  ```bash
  python ddqn_transformer_gameplayv9OptunaPrioReplayNoisy.py
  ```
- **LSTM Model**:
  ```bash
  python ddqn_lstm_gameplayv9OptunaPrioReplayNoisy.py
  ```
- **Random Gameplay**:
  ```bash
  python random_gameplay.py
  ```

### Analyze Result Metrics

1. **Metrics collected from the implementation:** To analyse the relevant metrics extracted from the Optuna trials go to the ```analysis``` folder. Here you can link the relevant
yaml files which contain the collected metrics from each trial into a Jupyter Notebook. Once linked you can create a set of standard plots showing 
loss and reward metrics as well as Q-value information.
2. **Metrics collected by Optuna** The setup automatically create an Optuna database which can be loaded into an Optuna dashboard.
To perform this activity to the folder ```output/optuna_study``` and perform the following steps:
    ```bash
      optuna-dashboard sqlite:///<optuna-study-db-name>.db
      ```


## License
This project is licensed under the terms of the MIT [LICENSE](LICENSE).
