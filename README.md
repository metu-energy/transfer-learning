# Introduction

This is the official repository for the paper "Transfer Learning for Heating Energy Consumption Prediction Using Urban Building Energy Models" (under review) by Ilkim Canli, Yusuf Meric Karadag, Sevval Ucar, Ismail Talaz, Fatma Ece Gursoy, Ipek Gursel Dino, Sinan Kalkan.

The repository is under construction. Soon, it will contain the code, the dataset and the machine learning models used in the paper. 

## Structure of the Repository

This repository contains the implementation of transfer learning experiments for heating energy consumption prediction. The file structure is explained as:

```
├── DATA/                      # Contains city-specific datasets 
│   ├── ANKARA/                # Ankara city data 
│   │   └── ankara.csv         
│   ├── ERZURUM/               # Erzurum city data 
│   │   └── erzurum.csv        
│   ├── IZMIR/                 # Izmir city data 
│   │   └── izmir.csv          
│   └── KAYSERI/               # Kayseri city data 
│       └── kayseri.csv        
├── ankara_source_model.pt     # Pre-trained source model for transfer learning
├── TL_experiment_framework.py # Core experiment framework implementation 
└── TL_experiment_runner.py    # Script to run the experiments
```


## Usage 

Install the required packages by running:
```bash
pip install -r requirements.txt
```

The main driver script is TL_experiment_runner.py. This script reads the city-specific datasets from the DATA folder, loads the pre-trained source model (if experiment type is not baseline), and runs the experiments. Experiment results are saved to WandB for ease of tracking and visualization. (As results are saved to WandB, you need to have a WandB account logged in your workspace.)


The TL_experiment_framework.py expects the following arguments:
- `--city`: City for the experiment. Choices are 'erzurum', 'izmir', 'kayseri'. (required)
- `--ratio`: Ratio of training data to use (0-1). (required)
- `--experiment_type`: Type of experiment to run. Choices are 'baseline', 'transfer_learning', 'lora', 'lora_plus'. (required)
- `--rank`: Rank for LoRA/LoRA+ experiments. (required for LoRA/LoRA+)
- `--lr_ratio`: Learning rate ratio for LoRA+ experiments. (required for LoRA+)
- `--seed`: Random seed. Default is 22. (optional)
- `--n_folds`: Number of folds for cross-validation. Default is 10. (optional)
- `--n_trials`: Number of optimization trials (experiments). Default is 40. (optional)
- `--n_epochs`: Number of training epochs. Default is 20. (optional)
- `--layer_num`: Number of layers in the model. Default is 4. (optional)
- `--layer_size`: Size of layers in the model. Default is 64. (optional)

To run different experiments, several examples are provided:
```bash
python TL_experiment_runner.py --city izmir --ratio 0.1 --experiment_type baseline
python TL_experiment_runner.py --city kayseri --ratio 0.3 --experiment_type transfer_learning
python TL_experiment_runner.py --city erzurum --ratio 0.5 --experiment_type lora --rank 2
python TL_experiment_runner.py --city izmir --ratio 1.0 --experiment_type lora_plus --rank 4 --lr_ratio 8
```

After training, the models are saved to their designated folders like `models/<city>/<experiment_type>/<ratio>` with filename containing the final validation and test R² scores appended with fold number and additional model information if available (rank variable of LoRA/LoRA+ and lr_ratio variable of LoRA+).