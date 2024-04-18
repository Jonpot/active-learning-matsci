# Models
from efficient_models_pytorch_3d import EfficientUnet3D, EfficientUnetPlusPlus3D

# Install packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.accelerators import find_usable_cuda_devices
from torchmetrics import R2Score
from kornia.enhance import histogram

from baal.utils.pytorch_lightning import ActiveLightningModule, BaaLDataModule, BaalTrainer, ResetCallback
from baal.active.dataset import ActiveLearningDataset
from baal.active.heuristics import Variance, CombineHeuristics

import static_frame as sf
from pathlib import Path
import zipfile
from zipfile import ZipFile
import numpy as np
from vedo import Volume, Text2D
from slicer import Slicer3DTwinPlotter
from tqdm import tqdm
import sys
import time
from datetime import datetime
import re
import copy

class NPZArchive ():
    def __init__(self, file, compress=True, allow_pickle=True, pickle_kwargs=None):
        # Setup NPZ File
        from numpy.lib.format import write_array
        from numpy.lib.npyio import zipfile_factory
        from numpy.compat import os_fspath

        self.allow_pickle = allow_pickle
        self.pickle_kwargs = pickle_kwargs
        self.write_array = write_array
        
        if not hasattr(file, 'write'):
            self.file = os_fspath(file)
            if not self.file.endswith('.npz'):
                self.file = file + '.npz'

        if compress:
            compression = zipfile.ZIP_LZMA
        else:
            compression = zipfile.ZIP_STORED

        # Open NPZ File
        self.zipf = zipfile_factory(self.file, mode="w", compression=compression)

    def write(self, vol, label):
        # Write Arrays to NPZ File
        filename = label + '.npy'

        # always force zip64, gh-10776
        with self.zipf.open(filename, 'w', force_zip64=True) as vol_file:
            self.write_array(vol_file, vol,
                             allow_pickle=self.allow_pickle,
                             pickle_kwargs=self.pickle_kwargs)

    def close(self):
        # Close NPZ File
        self.zipf.close()


class SurrogateDataset(Dataset):
    def __init__(self, dataset_info, df, transform=None):
        # Initialize Parameters
        self.array_labels = df['array_label'].values
        self.input_npzs = df['128_npz'].values
        self.predict = dataset_info['predict']
        
        self.stress_npzs = df['mean_stress_field_npz'].values
        self.strain_npzs = df['mean_strain_field_npz'].values
        
        self.num_channels = dataset_info['num_channels']

    def __getitem__(self, index):
        # Get Input Volume
        with ZipFile(self.input_npzs[index], 'r') as input_npz_file:
            x_vol = np.load(input_npz_file.open(f"{self.array_labels[index]}.npy"))
        
        # Get Stress Volume
        with ZipFile(self.stress_npzs[index], 'r') as stress_npz_file:
            stress_field = np.load(stress_npz_file.open(f"{self.array_labels[index]}.npy"))

        # Get Strain Volume
        with ZipFile(self.strain_npzs[index], 'r') as strain_npz_file:
            strain_field = 1e5 * np.load(strain_npz_file.open(f"{self.array_labels[index]}.npy"))

        # Prepare Volumes for PyTorch
        if self.num_channels == 1:
            x_vol = torch.Tensor(x_vol).unsqueeze(0)
            stress_field = torch.Tensor(stress_field).unsqueeze(0)
            strain_field = torch.Tensor(strain_field).unsqueeze(0)
        elif self.num_channels == 3:
            x_vol = torch.Tensor(x_vol).unsqueeze(0).repeat(3,1,1,1)
            stress_field = torch.Tensor(stress_field).unsqueeze(0).repeat(3,1,1,1)
            strain_field = torch.Tensor(strain_field).unsqueeze(0).repeat(3,1,1,1)

        # Return Volumes and Label
        return x_vol, (stress_field, strain_field)

    def __len__(self):
        # Get Length of Dataset
        return len(self.array_labels)


class SurrogateDataModule(BaaLDataModule):
    def __init__(self, dataset_info, df, batch_size):
        # Initialize Parameters
        self.dataset_info = dataset_info
        self.df = df.to_pandas()
        initial_train_size = dataset_info['initial_train_size']
        val_size = dataset_info['val_size']
        test_size = dataset_info['test_size']

        self.num_workers = dataset_info['num_workers']
        self.persistent_workers = dataset_info['persistent_workers']
        batch_size = batch_size

        # Split Data
        train_df, self.test_df = train_test_split(self.df, test_size=test_size,
                                                  stratify=self.df['cluster_label'],
                                                  shuffle=True, random_state=42)
        
        active_df, self.val_df = train_test_split(train_df, test_size=val_size,
                                                       stratify=train_df['cluster_label'],
                                                       shuffle=True, random_state=42)
        self.active_df = active_df.reset_index(drop=True)
        
        # Initialize Datasets
        active_dataset = ActiveLearningDataset(SurrogateDataset(self.dataset_info, self.active_df))
        self.validation_dataset = SurrogateDataset(self.dataset_info, self.val_df)
        self.test_dataset = SurrogateDataset(self.dataset_info, self.test_df)

        # Label Initial Training Data
        label_df, _ = train_test_split(self.active_df, train_size=initial_train_size,
                                       stratify=self.active_df['cluster_label'],
                                       random_state=42)
        label_indices = label_df.index.values
        active_dataset.label(label_indices)

        super().__init__(active_dataset=active_dataset, batch_size=batch_size)

    def train_dataloader(self):
        return DataLoader(self.active_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers) # persistent_workers=True for DDP

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers) # persistent_workers=True for DDP

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)
            

class SurrogateModule(ActiveLightningModule):
    def __init__(self, model_info, hyperparameters, plot=False):
        super(SurrogateModule, self).__init__()
        # Initialize Parameters
        self.name = model_info['name']
        self.encoder_ckpt = model_info['encoder_ckpt']
        self.freeze_encoder = model_info['freeze_encoder']
        self.decoder = model_info['decoder']
        self.decoder_attention = model_info['decoder_attention']
        self.dropout = model_info['dropout']
        self.num_channels = model_info['num_channels']
        self.num_classes = model_info['num_classes']
        self.plot = plot

        self.model = self.get_model()
    
        self.optimizer_name = hyperparameters['optimizer']
        self.batch_size = hyperparameters['batch_size']
        self.lr = hyperparameters['lr']
        self.sch_factor = hyperparameters['sch_factor']
        self.sch_patience = hyperparameters['sch_patience']
        self.mc_iterations = hyperparameters['mc_iterations']
        
        self.loss = hyperparameters['loss']
        self.score = hyperparameters['score']

    def get_model(self):
        if self.decoder == 'unet':
            model = EfficientUnet3D(self.name,
                                    encoder_weights=self.encoder_ckpt,
                                    freeze_encoder=self.freeze_encoder,
                                    decoder_attention_type=self.decoder_attention,
                                    in_channels=self.num_channels,
                                    classes=self.num_classes,
                                    dropout=self.dropout)
        
        elif self.decoder == 'unet++':
            model = EfficientUnetPlusPlus3D(self.name,
                                            encoder_weights=self.encoder_ckpt,
                                            freeze_encoder=self.freeze_encoder,
                                            in_channels=self.num_channels,
                                            classes=self.num_classes)
        
        else: print("Invalid decoder choice"); sys.exit()
        
        return model
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, (stress_field, strain_field) = batch
        masks = self(x)
        
        stress_mask = masks[:,0,:,:,:].unsqueeze(1)
        stress_loss = self.loss(stress_mask, stress_field)
        stress_score = self.score(stress_mask.flatten(), stress_field.flatten())

        strain_mask = masks[:,1,:,:,:].unsqueeze(1)
        strain_loss = self.loss(strain_mask, strain_field)
        strain_score = self.score(strain_mask.flatten(), strain_field.flatten())
        
        train_loss = stress_loss + strain_loss
        train_score = (stress_score + strain_score) / 2

        if self.plot:
            icon = x.squeeze().int().cpu().detach().numpy()
            
            stress_field = stress_field.squeeze().int().cpu().detach().numpy()
            stress_mask = stress_mask.squeeze().int().cpu().detach().numpy()
            self.plot_fields(stress_field, stress_mask, icon)

            strain_field = strain_field.squeeze().int().cpu().detach().numpy()
            strain_mask = strain_mask.squeeze().int().cpu().detach().numpy()
            self.plot_fields(strain_field, strain_mask, icon)

        self.log_dict({f"train_loss({self.loss.__class__.__name__})": train_loss,
                       f"train_score({self.score.__class__.__name__})": train_score},
                       batch_size=self.batch_size, on_epoch=True, on_step=False, sync_dist=True)
        
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, (stress_field, strain_field) = batch
        masks = self(x)
        
        stress_mask = masks[:,0,:,:,:].unsqueeze(1)
        stress_loss = self.loss(stress_mask, stress_field)
        stress_score = self.score(stress_mask.flatten(), stress_field.flatten())

        strain_mask = masks[:,1,:,:,:].unsqueeze(1)
        strain_loss = self.loss(strain_mask, strain_field)
        strain_score = self.score(strain_mask.flatten(), strain_field.flatten())
        
        val_loss = stress_loss + strain_loss
        val_score = (stress_score + strain_score) / 2

        if self.plot:
            icon = x.squeeze().int().cpu().detach().numpy()
            
            stress_field = stress_field.squeeze().int().cpu().detach().numpy()
            stress_mask = stress_mask.squeeze().int().cpu().detach().numpy()
            self.plot_fields(stress_field, stress_mask, icon)

            strain_field = strain_field.squeeze().int().cpu().detach().numpy()
            strain_mask = strain_mask.squeeze().int().cpu().detach().numpy()
            self.plot_fields(strain_field, strain_mask, icon)

        self.log_dict({f"val_loss({self.loss.__class__.__name__})": val_loss,
                       f"val_score({self.score.__class__.__name__})": val_score},
                       batch_size=self.batch_size, on_epoch=True, on_step=False, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, (stress_field, strain_field) = batch
        masks = self(x)
        
        stress_mask = masks[:,0,:,:,:].unsqueeze(1)
        stress_loss = self.loss(stress_mask, stress_field)
        stress_score = self.score(stress_mask.flatten(), stress_field.flatten())

        strain_mask = masks[:,1,:,:,:].unsqueeze(1)
        strain_loss = self.loss(strain_mask, strain_field)
        strain_score = self.score(strain_mask.flatten(), strain_field.flatten())
        
        test_loss = stress_loss + strain_loss
        test_score = (stress_score + strain_score) / 2

        if self.plot:
            icon = x.squeeze().int().cpu().detach().numpy()

            stress_field = stress_field.squeeze().int().cpu().detach().numpy()
            stress_mask = stress_mask.squeeze().int().cpu().detach().numpy()
            self.plot_fields(stress_field, stress_mask, icon)

            strain_field = strain_field.squeeze().int().cpu().detach().numpy()
            strain_mask = strain_mask.squeeze().int().cpu().detach().numpy()
            self.plot_fields(strain_field, strain_mask, icon)
        
        self.log_dict({f"test_loss({self.loss.__class__.__name__})": test_loss,
                       f"test_score({self.score.__class__.__name__})": test_score},
                       batch_size=1, sync_dist=True)
        
    def predict_step(self, batch, batch_idx):
        # Enable Dropout Layers
        self.enable_dropout(self.model)

        # Get Input Volume
        x, _ = batch

        # Initialize lists to store the maximum values for each Monte Carlo iteration
        stress_max_values = []
        strain_max_values = []
        for _ in range(self.mc_iterations):
            # Get the model's predictions
            masks = self(x)

            # Separate the stress and strain fields and add a dimension
            stress_mask = masks[:, 0, :, :, :].unsqueeze(1)
            strain_mask = masks[:, 1, :, :, :].unsqueeze(1)

            # Get the maximum values for each instance in the batch
            max_stress = torch.amax(stress_mask, dim=(2, 3, 4))
            max_strain = torch.amax(strain_mask, dim=(2, 3, 4))

            # Add the maximum values for this iteration to the lists
            stress_max_values.append(max_stress)
            strain_max_values.append(max_strain)

        # Convert the lists of maximum values to tensors
        # The resulting tensors should have the shape [mc_iterations, batch_size, 1]
        stress_max_values = torch.stack(stress_max_values)
        strain_max_values = torch.stack(strain_max_values)

        # Transpose the tensors to get the shape [batch_size, 1, mc_iterations]
        stress_max_values = stress_max_values.transpose(0, 1)
        strain_max_values = strain_max_values.transpose(0, 1)

        return [stress_max_values, strain_max_values]

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optimizer_name)(self.model.parameters(), lr=self.lr)
        lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer, mode='min',
                                     factor=self.sch_factor, patience=self.sch_patience),
                        'monitor': f"val_loss({self.loss.__class__.__name__})"}
        return [optimizer], [lr_scheduler]
    
    @staticmethod
    def enable_dropout(model):
        # Eable the dropout layers during during prediction for Monte-Carlo Dropout
        model.train()  # Set the whole model to training mode
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm3d):
                m.eval()  # Set BatchNorm3d layers to evaluation mode

    @staticmethod
    def plot_fields(target, mask, icon):
        # Create vedo Volume
        vol1 =  Volume(target.astype('f4'))
        vol2 =  Volume(mask.astype('f4'))
        icon = Volume(icon).alpha((0, 1))
        
        # Plot Stress Fields
        target_min = np.min(target)
        target_max = np.max(target)
        target_average = np.average(target)
        target_scf = target_max / target_average

        mask_min = np.min(mask)
        mask_max = np.max(mask)
        mask_average = np.average(mask)
        mask_scf = mask_max / mask_average

        print("\n######################################")
        print(f"Target Min = {target_min}, Mask Min = {mask_min}, Difference = {np.abs(target_min - mask_min)}")
        print(f"Target Max = {target_max}, Mask Max = {mask_max}, Difference = {np.abs(target_max - mask_max)}")
        print(f"Target Average = {target_average}, Mask Average = {mask_average}, Difference = {np.abs(target_average - mask_average)}")
        print(f"Target SCF = {target_scf}, Mask SCF = {mask_scf}, Difference = {np.abs(target_scf - mask_scf)}")
        print("######################################\n")

        plt = Slicer3DTwinPlotter(
            vol1, vol2, icon,
            shape=(1, 2),
            size = (1800, 950),
            sharecam=True,
            bg="white", 
            bg2="lightblue")

        plt.at(0).add(Text2D("Target",
                                s=1.2,
                                pos="top-center"))
        
        plt.at(0).add(Text2D(f"Min = {target_min}, Mean = {target_average:.3f}, Max = {target_max}, SCF = {target_scf:.3f}",
                                s=0.8,
                                bg='grey',
                                alpha=0.5,
                                pos='bottom-right'))
        
        plt.at(1).add(Text2D("Prediction",
                                s=1.2,
                                pos="top-center"))
        
        plt.at(1).add(Text2D(f"Min = {mask_min}, Mean = {mask_average:.3f}, Max = {mask_max}, SCF = {mask_scf:.3f}",
                                s=0.8,
                                bg='grey',
                                alpha=0.5,
                                pos='bottom-right'))
        
        plt.show(viewup='z')
        plt.at(0).reset_camera()
        plt.interactive().close()


class Model():
    def __init__(self, dataset_info, model_info, hyperparameters):
        super(Model, self).__init__()
        # Initialize Parameters
        self.dataset_info = dataset_info
        self.dataset = self.dataset_info['dataset']
        self.data_dir = self.dataset_info['data_dir']
        self.dfs_zip = dataset_info['dfs_zip']
        self.samples_dir = dataset_info['samples_dir']
        self.sample_nums = dataset_info['sample_nums']
        self.include = dataset_info['include']
        self.npz_size = dataset_info['npz_size']
        self.plot = dataset_info['plot']

        self.model_info = model_info
        self.surrogate_ckpt = self.model_info['surrogate_ckpt']

        training_info = self.model_info['training_info']
        self.distributed = training_info['distributed']
        self.max_epochs = training_info['max_epochs']
        self.mixed = training_info['mixed_precision']
        self.early_stopping = training_info['early_stopping']

        self.hyperparameters = hyperparameters
        self.heuristic = self.hyperparameters['heuristic']
        self.query_size = self.hyperparameters['query_size']
        self.al_steps = self.hyperparameters['al_steps']

        self.loss = self.hyperparameters['loss']
        self.score = self.hyperparameters['score']
        self.batch_size = self.hyperparameters['batch_size']

        self.setup()

    def setup(self):
        self.model_args = {'model_info': self.model_info, 'hyperparameters': self.hyperparameters, 'plot': self.plot}

        seed_everything(42, workers=True)
        torch.set_float32_matmul_precision('medium')
        
        # Load Bus
        self.bus = sf.Bus.from_zip_npz(self.dfs_zip)
        self.ct_scans = list(self.bus.keys())
        
        # Select CT Scans
        if self.include != 'all':
            scan_indexes = [i for i in range(len(self.ct_scans)) if self.ct_scans[i] in self.include]
            self.bus = self.bus.iloc[scan_indexes]
            self.ct_scans = list(self.bus.keys())
        
        # Select Columns and Load Dataframe
        self.columns = ['array_label', '128_npz', 'mean_stress_field_npz', 'mean_strain_field_npz', 'cluster_label']
        dfs = []
        for sample_num in self.sample_nums:
            sample_dir = self.samples_dir / f"sample_{sample_num}"
            file_label = f"{self.dataset}_massif_sample_{sample_num}"
            sample_csv_file = sample_dir / f"{file_label}.csv"

            df = sf.Frame.from_csv(sample_csv_file, encoding='utf8', name=f"sample_{sample_num}")
            df = df.drop[['__index0__']]
            dfs.append(df)
        df = sf.Quilt.from_frames(dfs, retain_labels=True)[self.columns]
      
        self.data_args = {'dataset_info': self.dataset_info, 'df': df, 'batch_size': self.batch_size}
       
    def active_learning(self):
        # Training Pipeline
        datamodule = SurrogateDataModule(**self.data_args)
        heuristic = CombineHeuristics(heuristics=[Variance() for _ in range(2)], weights=[0.5, 0.5])

        if self.surrogate_ckpt:
            model = SurrogateModule.load_from_checkpoint(self.surrogate_ckpt, **self.model_args,)
        else:
            model = SurrogateModule(**self.model_args)

        logger = TensorBoardLogger(f"surrogate_model_dir/train_logs", name=f"{self.dataset}",
                                   default_hp_metric=False)

        if self.distributed and torch.cuda.is_available():
            strategy = DDPStrategy(find_unused_parameters=True)
        else:
            strategy = 'auto'

        checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor=f"val_loss({self.loss.__class__.__name__})", mode='min')
        early_stop_callback = EarlyStopping(monitor=f"val_loss({self.loss.__class__.__name__})", mode='min')
        reset_weights_callback = ResetCallback(copy.deepcopy(model.state_dict()))

        callbacks = [checkpoint_callback, reset_weights_callback]
        if self.early_stopping:
            callbacks.append(early_stop_callback)

        trainer = BaalTrainer(
            logger=logger,
            max_epochs=self.max_epochs,
            enable_checkpointing=True,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            strategy=strategy,
            devices=find_usable_cuda_devices() if torch.cuda.is_available() else 1,
            precision='16-mixed' if self.mixed else 32,
            callbacks=callbacks,
            dataset=datamodule.active_dataset,
            heuristic=heuristic,
            query_size=self.query_size)
        
        for al_step in range(self.al_steps):
            print(f"Step: {al_step}, Dataset Size: {len(datamodule.active_dataset)}")

            trainer.fit(model, datamodule=datamodule)  # Train the model on the labelled set.
        
            best_model_path = Path(checkpoint_callback.best_model_path)
            best_epoch_search = re.search('epoch=(.*)-', best_model_path.as_posix())
            best_epoch = int(best_epoch_search.group(1))
            
            best_val_score = checkpoint_callback.best_model_score.item()
            best_test_score = trainer.test(ckpt_path='best', datamodule = datamodule)[0][f"test_score({self.score.__class__.__name__})"]
        
            metrics = {'best_epoch': best_epoch, 'val_score': best_val_score, 'test_score': best_test_score}
            trainer.logger.log_hyperparams(self.hyperparameters, metrics)

            should_continue = trainer.step(model, datamodule=datamodule)  # Label the top-k most uncertain examples.
            
            if not should_continue:
                break

    def test(self):
        # Testing Pipeline
        datamodule = SurrogateDataModule(**self.data_args)

        if self.surrogate_ckpt:
            model = SurrogateModule.load_from_checkpoint(self.surrogate_ckpt, **self.model_args,)
        else:
            print("No surrogate model checkpoint found. Need checkpoint for testing"); sys.exit()

        logger = TensorBoardLogger(f"surrogate_model_dir/test_logs",
                                   name=f"{self.dataset}",
                                   default_hp_metric=False)

        trainer = Trainer(
            logger=logger,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            num_nodes=1)
        
        trainer.test(model, datamodule = datamodule)
    

def main():
    # TODO: Params loader
    # Arguments
    dataset = 'ct_scans'
    massif_sample_nums = (0,) # Which MASSIF samples to use with fit and test
    include = 'all' # 'all' or ('TTT-AM-P-1-62',) to use with get_stress_fields
    npz_size = 1000 # Number of volumes per npz container
    num_workers = 12 # Select number of CPU resources
    plot_volumes = False # True or False

    task = 'active_learning' # 'active_learning', 'test'
    model_type = '3D' # 2D or 3D
    model_name = 'efficientnet-b0' # '*-b0', '*-b1', '*-b2', '*-b3', '*-b4', '*-b5', '*-b6', or '*-b7'
    encoder_ckpt = None # 'autoencoder/ae_pretraining_ct_scans_all.ckpt' or None
    freeze_encoder = False
    decoder = 'unet' # 'unet' or 'unet++'
    decoder_attention = 'scse' # None, 'se', or 'scse'
    decoder_dropout = 0.2 # None or float
    head_dropout = 0.1 # None or float
    num_channels = 1 # 1 or 3
    num_classes = 2 # int (number of output masks)
    surrogate_ckpt = None # 'surrogate_model_dir/unet_scse_pretraining_ct_scans_sample_0.ckpt' or None
    
    # Training Details
    max_epochs = 1
    distributed = True
    persistent_workers = True if distributed else False
    early_stopping = False
    mixed_precision = True
    
    training_info = {'distributed': distributed, 'max_epochs': max_epochs,
                        'mixed_precision': mixed_precision, 'early_stopping': early_stopping}

    # Active Learning Hyperparameters
    heuristic = 'variance' # 'variance', 'random'
    initial_train_size = 750 # 2500 total samples: 750 labeled, 1000 unlabeled, val_size=500 and test_size=250
    val_size = 500
    test_size = 250
    active_learning_steps = 10
    mc_sampling_iterations = 3
    query_size = 100  # Total queries = query_size * active_learning_steps = 25 * 40 = 1000

    # Model Hyperparameters
    loss = MSELoss()
    score = R2Score()
    optimizer_name = 'AdamW'
    batch_size = 1
    lr = 0.0022980
    sch_factor = 0.7
    sch_patience = 0

    # File Setup
    data_dir = Path(f"datasets/{dataset}")
    dfs_zip = data_dir / f"{data_dir.name}_dfs.zip"
    samples_dir = Path(f"surrogate_model_dir/massif_samples/{dataset}")
    
    if task == 'get_stress_fields': predict = True
    else: predict = False
    
    # Dictionary Setup
    dataset_info = {'dataset': dataset, 'data_dir': data_dir, 'dfs_zip': dfs_zip,
                    'initial_train_size': initial_train_size, 'val_size': val_size, 'test_size': test_size,
                    'samples_dir': samples_dir, 'sample_nums': massif_sample_nums, 'include': include,
                    'predict': predict, 'npz_size': npz_size, 'plot': plot_volumes,
                    'num_workers': num_workers, 'num_channels': num_channels,
                    'persistent_workers': persistent_workers if persistent_workers else None}
    
    model_info = {'task': task, 'model_type': model_type, 'name': model_name, 'encoder_ckpt': encoder_ckpt,
                  'freeze_encoder': freeze_encoder, 'decoder': decoder, 'decoder_attention': decoder_attention, 
                  'dropout': {'decoder_dropout': decoder_dropout, 'head_dropout': head_dropout},
                  'num_channels': num_channels,'num_classes': num_classes,
                  'surrogate_ckpt': surrogate_ckpt, 'training_info': training_info}
    
    hyperparameters = {'heuristic': heuristic, 'al_steps': active_learning_steps,
                       'mc_iterations': mc_sampling_iterations, 'query_size': query_size, 'loss': loss, 
                       'score': score, 'optimizer': optimizer_name, 'batch_size': batch_size,'lr': lr,
                       'sch_factor': sch_factor, 'sch_patience': sch_patience}
    
    # Main
    model = Model(dataset_info, model_info, hyperparameters)
    if task == 'active_learning': model.active_learning()
    elif task == 'test': model.test()
    else: print("Invalid task"); sys.exit()
    

if __name__ == "__main__":
    # TODO: Params loader
    start_time = time.time()

    main()

    duration = round(((time.time() - start_time))/60, 2)

    if duration >= 60:
        duration = f"{round(((time.time() - start_time))/3600, 2)} hours"
    else:
        duration = f"{round(((time.time() - start_time))/60, 2)} minutes"

    timestamp = datetime.now().strftime("%m-%d-%Y at %H:%M:%S")

    print(f"\nFinished in {duration} on {timestamp}")
    