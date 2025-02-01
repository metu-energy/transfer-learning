# experiment_base.py
from abc import ABC, abstractmethod
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import wandb
from sklearn import preprocessing
from copy import deepcopy
import loralib as lora
from dataclasses import dataclass
import os

@dataclass
class ExperimentConfig:
    city: str
    ratio: float
    experiment_type: str
    rank: int = None
    lr_ratio: float = None
    seed: int = 22
    n_folds: int = 10
    n_trials: int = 40
    n_epochs: int = 20
    layer_num: int = 4
    layer_size: int = 64
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

class dataset_gen(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(np.array(X.values, dtype=np.float32))
        self.y = torch.from_numpy(np.array(y.values, dtype=np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        return self.X[index, :], self.y[index, :]

class BaseExperiment(ABC):
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.seed_everything()

        # Initialize scalers
        self.SCALER_HEAT = None
        self.SCALERS = {}

        # Features list
        self.heat_features = [
            'formFactor', 'AspectRatio', 'wwr_N', 'wwr_W', 'wwr_S', 'wwr_E',
            'SE_N', 'SE_W', 'SE_S', 'SE_E', 'T_heating', 'NumberOfPeople', 
            'LPD', 'EPD', 'u_Wall', 'u_Window', 'u_Roof', 'u_GroundFloor', 
            'SHGC', 'Infiltration', 'COP_boiler', 'verticalPos', 'Q_heating'
        ]

    def seed_everything(self):
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def weather_parameters(self, df, city):
        weather_data = {
            "izmir": {
                "Annual Sum Heating Degree Days": 1344.825,
                "Annual Average Dry Bulb Temperature": 17.84,
                "Annual Average of Global Horizontal Irradiation": 198.27
            },
            "ankara": {
                "Annual Sum Heating Degree Days": 2499.4,
                "Annual Average Dry Bulb Temperature": 13.37,
                "Annual Average of Global Horizontal Irradiation": 197.88
            },
            "erzurum": {
                "Annual Sum Heating Degree Days": 4732.650,
                "Annual Average Dry Bulb Temperature": 6.28,
                "Annual Average of Global Horizontal Irradiation": 179.99
            },
            "kayseri":
            {
                "Annual Sum Heating Degree Days": 3260.7,
                "Annual Average Dry Bulb Temperature": 10.62,
                "Annual Average of Global Horizontal Irradiation": 190.37
            }
        }

        if city.lower() in weather_data:
            for key, value in weather_data[city.lower()].items():
                df[key] = value

        return df

    def ankara_scalers(self):
        df_ankara = pd.read_csv("./DATA/ANKARA/ankara.csv" ,usecols=self.heat_features)
        df_ankara = self.weather_parameters(df_ankara, "ankara")

        self.SCALER_HEAT = preprocessing.StandardScaler()

        for ind in df_ankara:
            if ind not in ["Q_heating", "IOD", "verticalPos"]:
                df_ankara[ind] = df_ankara[ind].astype(float)
                scaler = preprocessing.StandardScaler()
                scaler = scaler.fit(np.array(df_ankara[ind]).reshape(-1, 1))
                df_ankara[ind] = scaler.transform(np.array(df_ankara[ind]).reshape(-1, 1))
                self.SCALERS[ind] = deepcopy(scaler)

        self.SCALER_HEAT = deepcopy(
            self.SCALER_HEAT.fit(np.array(df_ankara["Q_heating"]).reshape(-1, 1))
        )

    def prepare_x_y(self, df):
        vertPos = pd.get_dummies(df["verticalPos"])
        df.drop("verticalPos", inplace=True, axis=1)

        df = self.weather_parameters(df, self.config.city)
        self.ankara_scalers()

        df["Q_heating"] = self.SCALER_HEAT.transform(
            np.array(df["Q_heating"]).reshape(-1, 1)
        )

        for ind in df:
            if ind != "Q_heating":
                df[ind] = self.SCALERS[ind].transform(
                    np.array(df[ind]).reshape(-1, 1)
                )

        y = df.iloc[:,[21]]
        df = pd.concat([df, vertPos], axis=1)
        df.drop("Q_heating", inplace=True, axis=1)

        return df, y

    def get_dataset(self):
        city_file = f"./DATA/{self.config.city.upper()}/{self.config.city.lower()}.csv"
        df = pd.read_csv(city_file, delimiter=",", usecols=self.heat_features)
        df = df.reindex(columns=self.heat_features)
        X, y = self.prepare_x_y(df)
        return dataset_gen(X, y)

    def get_train_test_split(self):
        generator = torch.Generator().manual_seed(self.config.seed)
        dataset = self.get_dataset()

        dataset_size = len(dataset)
        train_size = int(0.85 * dataset_size)
        test_size = dataset_size - train_size

        train_set, test_set = random_split(
            dataset, [train_size, test_size], generator=generator
        )

        if self.config.ratio < 1.0:
            train_val_len = len(train_set)
            reduced_len = int(train_val_len * self.config.ratio)
            useless_len = train_val_len - reduced_len
            train_set, _ = random_split(
                train_set, [reduced_len, useless_len], generator=generator
            )

        return train_set, test_set

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def create_optimizer(self, model, lr, weight_decay):
        pass

    def train_epoch(self, model, train_loader, optimizer, criterion):
        model.train()
        train_outs, train_labels = [], []
        
        for inputs, labels in train_loader:
            inputs = inputs.to(self.device).float()
            labels = labels.to(self.device).float()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.flatten(), labels.flatten())
            loss.backward()
            optimizer.step()
            
            train_outs.extend(outputs.cpu().detach().numpy().flatten())
            train_labels.extend(labels.cpu().detach().numpy().flatten())
            
        return train_outs, train_labels

    def evaluate(self, model, data_loader):
        model.eval()
        outs, labels = [], []
        
        with torch.no_grad():
            for inputs, target in data_loader:
                inputs = inputs.to(self.device).float()
                target = target.to(self.device).float()
                
                outputs = model(inputs)
                
                outs.extend(outputs.cpu().detach().numpy().flatten())
                labels.extend(target.cpu().detach().numpy().flatten())
                
        return outs, labels

    def run_fold(self, fold_num, train_idx, val_idx, test_loader, train_set, lr, batch_size, weight_decay):
        model = self.create_model().to(self.device).float()
        optimizer = self.create_optimizer(model, lr, weight_decay)
        criterion = torch.nn.MSELoss()

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_idx),
            drop_last=False
        )

        val_loader = DataLoader(
            dataset=train_set,
            batch_size=len(val_idx),
            sampler=torch.utils.data.SubsetRandomSampler(val_idx),
            drop_last=False
        )

        for epoch in range(self.config.n_epochs):
            train_outs, train_labels = self.train_epoch(
                model, train_loader, optimizer, criterion
            )
            val_outs, val_labels = self.evaluate(model, val_loader)
            
            train_r2 = r2_score(train_labels, train_outs)
            val_r2 = r2_score(val_labels, val_outs)
            
            step = fold_num * self.config.n_epochs + epoch
            
            wandb.log({
                'step': step,
                f'fold_{fold_num}/train_r2': train_r2,
                f'fold_{fold_num}/val_r2': val_r2,
                f'fold_{fold_num}/train_loss': criterion(torch.tensor(train_outs), 
                                                    torch.tensor(train_labels)).item(),
                f'fold_{fold_num}/val_loss': criterion(torch.tensor(val_outs), 
                                                    torch.tensor(val_labels)).item(),
                'current_fold': fold_num,
                'current_epoch': epoch,
            })

        test_outs, test_labels = self.evaluate(model, test_loader)
        final_train_outs, final_train_labels = self.evaluate(model, train_loader)
        
        final_train_r2 = r2_score(final_train_labels, final_train_outs)
        final_val_r2 = r2_score(val_labels, val_outs)
        final_test_r2 = r2_score(test_labels, test_outs)

        wandb.log({
            f'fold_{fold_num}/final_train_r2': final_train_r2,
            f'fold_{fold_num}/final_val_r2': final_val_r2,
            f'fold_{fold_num}/final_test_r2': final_test_r2,
            'current_fold': fold_num,
        })

        save_dir = f"./models/{self.config.city}"
        os.makedirs(save_dir, exist_ok=True)

        base_filename = f"{self.config.experiment_type}_valr2_{final_val_r2:.4f}_testr2_{final_test_r2:.4f}"
        
        if self.config.experiment_type == "lora":
            base_filename += f"_rank_{self.config.rank}"
        elif self.config.experiment_type == "lora_plus":
            base_filename += f"_rank_{self.config.rank}_lrratio_{self.config.lr_ratio}"
        
        base_filename += f"_fold_{fold_num}"
        
        save_path = os.path.join(save_dir, f"{base_filename}.pt")

        # Save model based on experiment type
        if self.config.experiment_type == "transfer_learning" or self.config.experiment_type == "baseline":
            torch.save(model.state_dict(), save_path)
        else:  # lora or lora_plus
            torch.save(lora.lora_state_dict(model), save_path)
        
        return final_train_r2, final_val_r2, final_test_r2

    def run_experiment(self):
        import optuna
        
        def objective(trial):
            # Start a new WandB run for this trial
            # Make sure you are logged in to wandb
            wandb.init(
                project="building-energy",
                config={
                    "trial_number": trial.number,
                    "city": self.config.city,
                    "ratio": self.config.ratio,
                    "experiment_type": self.config.experiment_type,
                    "rank": self.config.rank,
                    "lr_ratio": self.config.lr_ratio,
                    "seed": self.config.seed,
                    "n_folds": self.config.n_folds,
                    "layer_num": self.config.layer_num,
                    "layer_size": self.config.layer_size
                },
                group=f"{self.config.city}_{self.config.experiment_type}_{self.config.ratio}",
                name=f"trial_{trial.number}",
                reinit=True
            )

            train_set, test_set = self.get_train_test_split()
            test_loader = DataLoader(
                test_set, batch_size=len(test_set), shuffle=True, drop_last=False
            )

            lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 48, 64])

            wandb.log({
                'learning_rate': lr,
                'weight_decay': weight_decay,
                'batch_size': batch_size
            })

            kfold = KFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.seed
            )
            
            r2_train_scores = []
            r2_val_scores = []
            r2_test_scores = [] 
            
            for fold_num, (train_idx, val_idx) in enumerate(kfold.split(train_set)):
                train_r2, val_r2, test_r2 = self.run_fold(
                    fold_num, train_idx, val_idx, test_loader, train_set, 
                    lr, batch_size, weight_decay
                )
                r2_train_scores.append(train_r2)
                r2_val_scores.append(val_r2)
                r2_test_scores.append(test_r2)
            
            mean_train_r2 = np.mean(r2_train_scores)
            std_train_r2 = np.std(r2_train_scores)
            
            mean_val_r2 = np.mean(r2_val_scores)
            std_val_r2 = np.std(r2_val_scores)
            
            mean_test_r2 = np.mean(r2_test_scores)
            std_test_r2 = np.std(r2_test_scores)
            
            wandb.log({
                'mean_val_r2': mean_val_r2,
                'std_val_r2': std_val_r2,
                'mean_train_r2': mean_train_r2,
                'mean_val_r2': mean_val_r2,
                'mean_test_r2': mean_test_r2,

            })
            
            wandb.finish()
            
            return mean_val_r2

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.n_trials)

class BaselineExperiment(BaseExperiment):
    def create_model(self):
        return Model(27, 1, self.config.layer_num, self.config.layer_size)

    def create_optimizer(self, model, lr, weight_decay):
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

class TransferLearningExperiment(BaseExperiment):
    def create_model(self):
        model = Model(27, 1, self.config.layer_num, self.config.layer_size)
        model.load_state_dict(torch.load("./ankara_source_model.pt"))
        return model

    def create_optimizer(self, model, lr, weight_decay):
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

class LoRAExperiment(BaseExperiment):
    def create_model(self):
        model = LoRAModel(
            27, 1, self.config.layer_num, self.config.layer_size, self.config.rank
        )
        lora.mark_only_lora_as_trainable(model)
        model.load_state_dict(torch.load("./ankara_source_model.pt"), strict=False)
        return model

    def create_optimizer(self, model, lr, weight_decay):
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

class LoRAPlusExperiment(BaseExperiment):
    def create_model(self):
        model = LoRAModel(
            27, 1, self.config.layer_num, self.config.layer_size, self.config.rank
        )
        lora.mark_only_lora_as_trainable(model)
        model.load_state_dict(torch.load("./ankara_source_model.pt"), strict=False)
        return model

    def create_optimizer(self, model, lr, weight_decay):
        from lora_plus import create_loraplus_optimizer
        optimizer_cls = torch.optim.AdamW
        optimizer_kwargs = {
            'lr': lr, 
            'eps': 1e-6, 
            'betas': (0.9, 0.999), 
            'weight_decay': weight_decay
        }
        return create_loraplus_optimizer(
            model, optimizer_cls, optimizer_kwargs, self.config.lr_ratio
        )

# Model classes
class Model(torch.nn.Module):
    def __init__(self, s_in=13, s_out=1, layer_num=32, layer_size=16):
        super().__init__()
        self.in_layer = torch.nn.Linear(s_in, layer_size)
        self.hidden_layers = torch.nn.ModuleList([
            torch.nn.Linear(layer_size, layer_size) for _ in range(layer_num-2)
        ])
        self.out1 = torch.nn.Linear(layer_size, 1)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.in_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        return self.out1(x).squeeze()

class LoRAModel(torch.nn.Module):
    def __init__(self, s_in=13, s_out=1, layer_num=32, layer_size=16, rank=4):
        super().__init__()
        self.in_layer = lora.Linear(s_in, layer_size, r=rank)
        self.hidden_layers = torch.nn.ModuleList([
            lora.Linear(layer_size, layer_size, r=rank) 
            for _ in range(layer_num-2)
        ])
        self.out1 = lora.Linear(layer_size, 1, r=rank)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.relu(self.in_layer(x))
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        return self.out1(x).squeeze()
