# run_experiment.py
import argparse
import wandb
from TL_experiment_framework import (
    ExperimentConfig,
    BaselineExperiment,
    TransferLearningExperiment,
    LoRAExperiment,
    LoRAPlusExperiment
)

def parse_args():
    parser = argparse.ArgumentParser(description='Run ML Experiments')
    
    # Required arguments
    parser.add_argument('--city', type=str, required=True, 
                       choices=['erzurum', 'izmir', 'kayseri'],
                       help='City for the experiment')
    parser.add_argument('--ratio', type=float, required=True,
                       help='Ratio of training data to use (0-1)')
    parser.add_argument('--experiment_type', type=str, required=True,
                       choices=['baseline', 'transfer_learning', 'lora', 'lora_plus'],
                       help='Type of experiment to run')
    
    # Optional arguments
    parser.add_argument('--rank', type=int,
                       help='Rank for LoRA/LoRA+ experiments')
    parser.add_argument('--lr_ratio', type=float,
                       help='Learning rate ratio for LoRA+ experiments')
    parser.add_argument('--seed', type=int, default=22,
                       help='Random seed')
    parser.add_argument('--n_folds', type=int, default=10,
                       help='Number of folds for cross-validation')
    parser.add_argument('--n_trials', type=int, default=40,
                       help='Number of optimization trials')
    parser.add_argument('--n_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--layer_num', type=int, default=4,
                       help='Number of layers in the model')
    parser.add_argument('--layer_size', type=int, default=64,
                       help='Size of layers in the model')
    
    args = parser.parse_args()
    
    # Validation
    if args.ratio <= 0 or args.ratio > 1:
        parser.error("Ratio must be between 0 and 1")
        
    if args.experiment_type in ['lora', 'lora_plus'] and args.rank is None:
        parser.error("Rank is required for LoRA and LoRA+ experiments")
        
    if args.experiment_type == 'lora_plus' and args.lr_ratio is None:
        parser.error("Learning rate ratio is required for LoRA+ experiments")
    
    return args

def main():
    args = parse_args()
    
    # Initialize wandb
    wandb.init(
        project="building-energy",
        config={
            "city": args.city,
            "ratio": args.ratio,
            "experiment_type": args.experiment_type,
            "rank": args.rank,
            "lr_ratio": args.lr_ratio,
            "seed": args.seed,
            "n_folds": args.n_folds,
            "n_trials": args.n_trials,
            "n_epochs": args.n_epochs,
            "layer_num": args.layer_num,
            "layer_size": args.layer_size
        }
    )
    
    # Create experiment config
    config = ExperimentConfig(
        city=args.city,
        ratio=args.ratio,
        experiment_type=args.experiment_type,
        rank=args.rank,
        lr_ratio=args.lr_ratio,
        seed=args.seed,
        n_folds=args.n_folds,
        n_trials=args.n_trials,
        n_epochs=args.n_epochs,
        layer_num=args.layer_num,
        layer_size=args.layer_size
    )
    
    # Create and run appropriate experiment
    if args.experiment_type == 'baseline':
        experiment = BaselineExperiment(config)
    elif args.experiment_type == 'transfer_learning':
        experiment = TransferLearningExperiment(config)
    elif args.experiment_type == 'lora':
        experiment = LoRAExperiment(config)
    elif args.experiment_type == 'lora_plus': 
        experiment = LoRAPlusExperiment(config)
    else:
        raise ValueError("Invalid experiment type")
        
    experiment.run_experiment()
    wandb.finish()

if __name__ == "__main__":
    main()
