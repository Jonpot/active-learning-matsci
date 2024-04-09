# Models
from efficient_models_pytorch_3d import EfficientUnet3D, EfficientUnetPlusPlus3D

# Install packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss, L1Loss
from torchmetrics import R2Score
from kornia.enhance import histogram
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.accelerators import find_usable_cuda_devices

import static_frame as sf
from pathlib import Path
import zipfile
from zipfile import ZipFile
import numpy as np
from vedo import Volume, Text2D, show
from slicer import Slicer3DPlotter, Slicer3DTwinPlotter
from tqdm import tqdm
import sys
import time
from datetime import datetime
import math
import re


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
        
        if self.predict:
            self.pore_normal_areas = df['pore_normal_area'].values
        else:
            self.stress_npzs = df['mean_stress_field_npz'].values
            self.strain_npzs = df['mean_strain_field_npz'].values
        
        self.num_channels = dataset_info['num_channels']
        self.transform = transform

    def __getitem__(self, index):
        # Get Input Volume
        with ZipFile(self.input_npzs[index], 'r') as input_npz_file:
            x_vol = np.load(input_npz_file.open(f"{self.array_labels[index]}.npy"))
        
        if not self.predict:
            # Get Stress Volume
            with ZipFile(self.stress_npzs[index], 'r') as stress_npz_file:
                stress_field = np.load(stress_npz_file.open(f"{self.array_labels[index]}.npy"))

            # Get Strain Volume
            with ZipFile(self.strain_npzs[index], 'r') as strain_npz_file:
                strain_field = 1e5 * np.load(strain_npz_file.open(f"{self.array_labels[index]}.npy"))

        # Prepare Volumes for PyTorch
        if self.num_channels == 1:
            x_vol = torch.Tensor(x_vol).unsqueeze(0)
            if not self.predict:
                stress_field = torch.Tensor(stress_field).unsqueeze(0)
                strain_field = torch.Tensor(strain_field).unsqueeze(0)
        elif self.num_channels == 3:
            x_vol = torch.Tensor(x_vol).unsqueeze(0).repeat(3,1,1,1)
            if not self.predict:
                stress_field = torch.Tensor(stress_field).unsqueeze(0).repeat(3,1,1,1)
                strain_field = torch.Tensor(strain_field).unsqueeze(0).repeat(3,1,1,1)
        
        # Apply Transformations
        if self.transform:
            x_vol = self.transform(x_vol)

        # Return Volumes and Label
        if self.predict:
            return x_vol, self.array_labels[index], self.pore_normal_areas[index]
        else:
            return x_vol, stress_field, strain_field

    def __len__(self):
        # Get Length of Dataset
        return len(self.array_labels)


class LitDataModule(LightningDataModule):
    def __init__(self, dataset_info, df, batch_size):
        super().__init__()
        # Initialize Parameters
        self.dataset_info = dataset_info
        self.df = df.to_pandas()

        self.num_workers = dataset_info['num_workers']
        self.persistent_workers = dataset_info['persistent_workers']
        self.batch_size = batch_size
        #self.prepare_data_per_node = True

    def setup(self, stage: str):
        train_df, self.test_df = train_test_split(self.df, test_size=0.1, stratify=self.df['ct_scan'],
                                                  shuffle=True, random_state=42)
        self.train_df, self.val_df = train_test_split(train_df, test_size=0.222, stratify=train_df['ct_scan'],
                                                      shuffle=True, random_state=42)
        
        if stage == 'fit':
            self.train_dataset = SurrogateDataset(self.dataset_info, self.train_df)
            self.validation_dataset = SurrogateDataset(self.dataset_info, self.val_df)

        if stage == "test":
            self.test_dataset = SurrogateDataset(self.dataset_info, self.test_df)

        if stage == 'predict':
            self.predict_dataset = SurrogateDataset(self.dataset_info, self.df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers, shuffle=True) # persistent_workers=True for DDP

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers) # persistent_workers=True for DDP

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=1, num_workers=self.num_workers)
    

class EarthMoversMSELoss(nn.Module):
    def __init__(self, bins=25, p=1, do_root=True, alpha=0.5):
        super(EarthMoversMSELoss, self).__init__()
        self.bins = bins
        self.p = p
        self.do_root = do_root
        self.alpha = alpha
        self.mae_loss = L1Loss()
        self.mse_loss = MSELoss()
        
    def forward(self, mask, target):
        mask = torch.flatten(mask, start_dim=1)
        mask_min = torch.min(mask)
        mask_max = torch.max(mask)

        target = torch.flatten(target, start_dim=1)
        target_min = torch.min(target)
        target_max = torch.max(target)

        min_val = torch.min(mask_min, target_min)
        max_val = torch.max(mask_max, target_max)
        bins = torch.linspace(min_val.item(), max_val.item(), self.bins).to(mask.device)

        mask_hist = histogram(mask, bins, bandwidth=torch.tensor(0.9))
        target_hist = histogram(target, bins, bandwidth=torch.tensor(0.9))
        
        #mae_loss = self.mae_loss(mask, target)
        mse_loss = self.mse_loss(mask, target)
        emd_loss = self.emd_loss(mask_hist, target_hist, p=self.p, do_root=self.do_root)
        #max_val_loss = self.mse_loss(mask_max, target_max)

        return (1 - self.alpha) * mse_loss + self.alpha * 1e5 * emd_loss# + 0.01 * max_val_loss
    
    def emd_loss(self, mask_hist, target_hist, p=1, do_root=True):
        # Compute the Earth Mover's Distance loss
        mask_cdf = torch.cumsum(mask_hist, dim=-1)
        target_cdf = torch.cumsum(target_hist, dim=-1)

        if p == 1: # Compute L1
            emd = self.mae_loss(mask_cdf, target_cdf)

        elif p == 2: # Compute L2
            emd = self.mse_loss(mask_cdf, target_cdf)
            if do_root:
                emd = torch.sqrt(emd)

        else: # Compute p-norm
            emd = torch.mean(torch.pow(torch.abs(mask_cdf - target_cdf), p), dim=-1)
            if do_root:
                emd = torch.pow(emd, 1 / p)

        return torch.mean(emd) # Mean of all batches


class PNormLoss(nn.Module):
    def __init__(self, weight=None, percentile=0.95, p=2, do_root=False):
        super(PNormLoss, self).__init__()
        self.weight = weight
        self.percentile = percentile
        self.p = p
        self.do_root = do_root
        
    def forward(self, mask, target):
        tensors = [mask, target]

        target_min = torch.min(target)
        target_max = torch.max(target)

        if self.weight:
            weights = torch.ones_like(target)
            weight_threshold = self.percentile * (target_max - target_min)
            weights[weights > weight_threshold] = self.weight
            tensors.append(weights)

        return self.p_norm_loss(*tensors, p=self.p, do_root=self.do_root)
    
    def p_norm_loss(self, mask, target, weights=None, p=1, do_root=False):
        if p == 1: # Compute L1
            loss = torch.abs(mask - target)
        else:
            loss = torch.pow(torch.abs(mask - target), p)

        if self.weight:
            loss = weights * loss

        p_norm = torch.mean(loss, dim=-1)

        if do_root:
            p_norm = torch.pow(p_norm, 1 / p)

        return torch.mean(p_norm) # Mean of all batches
            

class SurrogateModule (LightningModule):
    def __init__(self, model_info, hyperparameters, plot=False):
        super(SurrogateModule, self).__init__()
        # Initialize Parameters
        self.task = model_info['task']
        self.name = model_info['name']
        self.encoder_ckpt = model_info['encoder_ckpt']
        self.freeze_encoder = model_info['freeze_encoder']
        self.decoder = model_info['decoder']
        self.decoder_attention = model_info['decoder_attention']
        self.num_channels = model_info['num_channels']
        self.num_classes = model_info['num_classes']
        self.plot = plot

        self.model = self.get_model()
    
        self.optimizer_name = hyperparameters['optimizer']
        self.batch_size = hyperparameters['batch_size']
        self.lr = hyperparameters['lr']
        self.sch_factor = hyperparameters['sch_factor']
        self.sch_patience = hyperparameters['sch_patience']
        
        self.loss = hyperparameters['loss']
        self.score = hyperparameters['score']

        if self.task == 'get_stress_fields':
            self.stress_npz_archive = model_info['stress_npz_archive']
            self.strain_npz_archive = model_info['strain_npz_archive']
            self.stress_fields_npzs = []
            self.strain_fields_npzs = []
            self.min_stresses = []
            self.max_stresses = []
            self.average_stresses = []
            self.stress_cfs = []
            self.min_strains = []
            self.max_strains = []
            self.average_strains = []
            self.strain_cfs = []
            self.sscfs = []

    def get_model(self):
        if self.decoder == 'unet':
            model = EfficientUnet3D(self.name,
                                    encoder_weights=self.encoder_ckpt,
                                    freeze_encoder=self.freeze_encoder,
                                    decoder_attention_type=self.decoder_attention,
                                    in_channels=self.num_channels,
                                    classes=self.num_classes)
        
        elif self.decoder == 'unet++':
            model = EfficientUnetPlusPlus3D(self.name,
                                            encoder_weights=self.encoder_ckpt,
                                            freeze_encoder=self.freeze_encoder,
                                            in_channels=self.num_channels,
                                            classes=self.num_classes)
        
        else: print("Invalid decoder choice"); sys.exit()
        
        return model

    def training_step(self, batch, batch_idx):
        x, stress_field, strain_field = batch
        masks = self.model(x)
        
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
        x, stress_field, strain_field = batch
        masks = self.model(x)
        
        stress_mask = masks[:,0,:,:,:]#.unsqueeze(1)
        stress_loss = self.loss(stress_mask, stress_field)
        stress_score = self.score(stress_mask.flatten(), stress_field.flatten())

        strain_mask = masks[:,1,:,:,:]#.unsqueeze(1)
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
        x, stress_field, strain_field = batch
        masks = self.model(x)
        
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
            #self.plot_fields(stress_field, stress_mask, icon)

            strain_field = strain_field.squeeze().int().cpu().detach().numpy()
            strain_mask = strain_mask.squeeze().int().cpu().detach().numpy()
            self.plot_volume(strain_field, field=True)
            self.plot_fields(strain_field, strain_mask, icon)
        
        self.log_dict({f"test_loss({self.loss.__class__.__name__})": test_loss,
                       f"test_score({self.score.__class__.__name__})": test_score},
                       batch_size=1, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        x, label, pore_normal_area = batch
        masks = self.model(x)

        # Stress Field Data
        stress_mask = masks[:,0,:,:,:].unsqueeze(1)
        mean_stress_field = stress_mask.squeeze().int().cpu().detach().numpy()
        self.stress_npz_archive.write(mean_stress_field, label[0])

        min_stress = np.min(mean_stress_field)
        max_stress = np.max(mean_stress_field)
        average_stress = np.average(mean_stress_field)
        stress_cf = max_stress / average_stress
        
        # Strain Field Data
        strain_mask = masks[:,1,:,:,:].unsqueeze(1)
        mean_strain_field = strain_mask.squeeze().int().cpu().detach().numpy()/1e5
        self.strain_npz_archive.write(mean_strain_field, label[0])

        min_strain = np.min(mean_strain_field)
        max_strain = np.max(mean_strain_field)
        average_strain = np.average(mean_strain_field)
        strain_cf = max_strain / average_strain

        # Stress-Strain Concentration Factor
        sscf = math.sqrt(stress_cf * strain_cf * math.sqrt(pore_normal_area))
        
        # Collect Field Data
        self.stress_fields_npzs.append(self.stress_npz_archive.file)
        self.strain_fields_npzs.append(self.strain_npz_archive.file)
        self.min_stresses.append(min_stress)
        self.max_stresses.append(max_stress)
        self.average_stresses.append(average_stress)
        self.stress_cfs.append(stress_cf)
        self.min_strains.append(min_strain)
        self.max_strains.append(max_strain)
        self.average_strains.append(average_strain)
        self.strain_cfs.append(strain_cf)
        self.sscfs.append(sscf)

        if self.plot:
            icon = x.squeeze().int().cpu().detach().numpy()
            self.plot_volume(stress_mask, field=True, icon=icon)

    def configure_optimizers(self):
        optimizer = getattr(optim, self.optimizer_name)(self.model.parameters(), lr=self.lr)
        lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer, mode='min',
                                     factor=self.sch_factor, patience=self.sch_patience),
                        'monitor': f"val_loss({self.loss.__class__.__name__})"}
        return [optimizer], [lr_scheduler]
    
    @staticmethod
    def plot_volume(mask, field=False, icon=None, interactive=True):
        # Create vedo Volume
        vol = Volume(mask.astype('f4'))
        
        # Plot Stress Fields
        if field == True:
            mask_min = np.min(mask)
            mask_max = np.max(mask)
            mask_average = np.average(mask)
            mask_scf = mask_max / mask_average

            print("\n######################################")
            print("Prediction")
            print(f"Min = {mask_min}")
            print(f"Max = {mask_max}")
            print(f"Average = {mask_average}")
            print(f"SCF = {mask_scf}")
            print("######################################\n")

            icon = Volume(icon).alpha((0, 1))#[20:-20,20:-20,20:-20])
    
            plt = Slicer3DPlotter(
                vol,
                icon=icon,
                icon_size=0.25,
                bg="white",
                bg2="lightblue",
                cmaps='jet',
                use_slider3d=False)
            
            plt.at(0).add(Text2D("Target",
                                 s=1.2,
                                 pos="top-center"))
            
            plt.at(0).add(Text2D(f"Min = {mask_min}, Mean = {mask_average:.3f}, Max = {mask_max}, SCF = {mask_scf:.3f}",
                                 s=0.8,
                                 bg='grey',
                                 alpha=0.5,
                                 pos=(0.28,0.1)))
            
            #plt.show(viewup='z')
            plt.at(0).reset_camera()
            plt.interactive()#.close()
        
        # Plot Volume
        else:
            show(vol, axes=1, interactive=interactive, viewup='z', new=True)

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
        self.task = model_info['task']
        self.surrogate_ckpt = self.model_info['surrogate_ckpt']

        self.training_info = self.model_info['training_info']
        if self.training_info:
            self.distributed = self.training_info['distributed']
            self.max_epochs = self.training_info['max_epochs']
            self.mixed = self.training_info['mixed_precision']
            self.early_stopping = self.training_info['early_stopping']

        self.hyperparameters = hyperparameters
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
        if self.task == 'get_stress_fields':
            self.columns = ['array_label', '128_npz', 'ct_scan', 'pore_normal_area']
            df = None
        else:
            self.columns = ['array_label', '128_npz', 'mean_stress_field_npz', 'mean_strain_field_npz', 'ct_scan']
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
       
    def fit(self):
        # Training Pipeline
        datamodule = LitDataModule(**self.data_args)

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

        callbacks = [checkpoint_callback]
        if self.early_stopping:
            callbacks.append(early_stop_callback)

        trainer = Trainer(
            logger=logger,
            max_epochs=self.max_epochs,
            enable_checkpointing=True,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            strategy=strategy,
            devices=find_usable_cuda_devices() if torch.cuda.is_available() else 1,
            precision='16-mixed' if self.mixed else 32,
            callbacks=callbacks)
        
        trainer.fit(model, datamodule=datamodule)
        
        best_model_path = Path(checkpoint_callback.best_model_path)
        best_epoch_search = re.search('epoch=(.*)-', best_model_path.as_posix())
        best_epoch = int(best_epoch_search.group(1))
        
        best_val_score = checkpoint_callback.best_model_score.item()
        best_test_score = trainer.test(ckpt_path='best', datamodule = datamodule)[0][f"test_score({self.score.__class__.__name__})"]
       
        metrics = {'best_epoch': best_epoch, 'val_score': best_val_score, 'test_score': best_test_score}
        trainer.logger.log_hyperparams(self.hyperparameters, metrics)

    def test(self):
        # Testing Pipeline
        datamodule = LitDataModule(**self.data_args)

        #model = SurrogateModule.load_from_checkpoint(self.surrogate_ckpt, **self.model_args)

        if self.surrogate_ckpt:
            model = SurrogateModule.load_from_checkpoint(self.surrogate_ckpt, **self.model_args,)
        else:
            model = SurrogateModule(**self.model_args)

        logger = TensorBoardLogger(f"surrogate_model_dir/test_logs",
                                   name=f"{self.dataset}",
                                   default_hp_metric=False)

        trainer = Trainer(
            logger=logger,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            num_nodes=1)
        
        trainer.test(model, datamodule = datamodule)

    def get_stress_fields(self):
        # Get Stress Field Pipeline
        dfs = []
        pbar_filler = "#################################################"
        for ct_scan in (pbar0 := tqdm(self.ct_scans, leave=False)):
            pbar0.set_description(f"\n{pbar_filler}\n\nGetting stress fields for {ct_scan}\n\n{pbar_filler}")
            # Setup File Info
            scan_dir = self.data_dir / ct_scan
            scan_df_csv = scan_dir / f"{ct_scan}_pores_df.csv"

            stress_fields_dir = scan_dir / 'fields' / 'mean_stress_fields'
            stress_fields_dir.mkdir(parents=True, exist_ok=True)

            strain_fields_dir = scan_dir / 'fields' / 'mean_strain_fields'
            strain_fields_dir.mkdir(exist_ok=True)

            # Select Columns and Load Into Frame
            df = self.bus[ct_scan]

            # Slice Up Dataframe by NPZ Size
            num_loops = math.ceil(len(df)/self.npz_size)
            stress_fields_npzs = []
            strain_fields_npzs = []
            min_stresses = []
            max_stresses = []
            average_stresses = []
            stress_cfs = []
            min_strains = []
            max_strains = []
            average_strains = []
            strain_cfs = []
            sscfs = []
            for loop_num in (pbar1 := tqdm(range(num_loops), position=7, leave=False)):
                pbar1.set_description(f"Writing Pores to NPZ {loop_num}")
                start_index = loop_num * self.npz_size
                stop_index = start_index + self.npz_size if loop_num+1 != num_loops else None

                df_slice = df.iloc[start_index:stop_index]
                self.data_args['df'] = df_slice
                datamodule = LitDataModule(**self.data_args)

                # Open NPZ Archives
                stress_fields_npz = stress_fields_dir / f"{ct_scan}_mean_stress_fields_{loop_num}.npz"
                stress_npz_archive = NPZArchive(stress_fields_npz, compress=True)
                self.model_args['model_info']['stress_npz_archive'] = stress_npz_archive

                strain_fields_npz = strain_fields_dir / f"{ct_scan}_mean_strain_fields_{loop_num}.npz"
                strain_npz_archive = NPZArchive(strain_fields_npz, compress=True)
                self.model_args['model_info']['strain_npz_archive'] = strain_npz_archive
                
                model = SurrogateModule.load_from_checkpoint(self.surrogate_ckpt, **self.model_args)

                # Get Stress Field Prediction and Write to NPZ
                trainer = Trainer(
                    logger=False,
                    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                    devices=1,
                    num_nodes=1)
                
                trainer.predict(model, datamodule=datamodule)

                stress_fields_npzs.extend(model.stress_fields_npzs)
                strain_fields_npzs.extend(model.strain_fields_npzs)
                min_stresses.extend(model.min_stresses)
                max_stresses.extend(model.max_stresses)
                average_stresses.extend(model.average_stresses)
                stress_cfs.extend(model.stress_cfs)
                min_strains.extend(model.min_strains)
                max_strains.extend(model.max_strains)
                average_strains.extend(model.average_strains)
                strain_cfs.extend(model.strain_cfs)
                sscfs.extend(model.sscfs)

                # Close NPZ Archive
                stress_npz_archive.close()
                strain_npz_archive.close()

            # Write NPZ Directory Info to File
            stress_fields_npzs = sf.Series(stress_fields_npzs, name='mean_stress_field_npz')
            strain_fields_npzs = sf.Series(strain_fields_npzs, name='mean_strain_field_npz')

            # Unpack Fields Data
            min_stresses = sf.Series(min_stresses, name='min_stress')
            max_stresses = sf.Series(max_stresses, name='max_stress')
            average_stresses = sf.Series(average_stresses, name='average_stress')
            stress_cfs = sf.Series(stress_cfs, name='stress_cf')

            min_strains = sf.Series(min_strains, name='min_strain')
            max_strains = sf.Series(max_strains, name='max_strain')
            average_strains = sf.Series(average_strains, name='average_strain')
            strain_cfs = sf.Series(strain_cfs, name='strain_cf')

            sscfs = sf.Series(sscfs, name='sscf')

            mean_fields_data = sf.Frame.from_concat((min_stresses, max_stresses, average_stresses, stress_cfs,
                                                     min_strains, max_strains, average_strains, strain_cfs,
                                                     sscfs), axis=1)
            
            # Add Stress Field Data to Dataframe
            if 'mean_stress_field_npz' in df.keys():
                df = df.drop[['mean_stress_field_npz', 'mean_strain_field_npz', 'sscf',
                              'min_stress', 'max_stress', 'average_stress', 'stress_cf',
                              'min_strain', 'max_strain', 'average_strain', 'strain_cf']]

            df = df.insert_after('feature_npz', stress_fields_npzs)
            df = df.insert_after('mean_stress_field_npz', strain_fields_npzs)
            df = df.insert_after(df.columns[-1], mean_fields_data)
            df.to_csv(scan_df_csv)
            dfs.append(df)

        # Remake Bus
        bus = sf.Bus.from_frames(dfs, name = f"{self.data_dir.name}")
        bus.to_zip_npz(self.dfs_zip)
    

def main():
    # TODO: Params loader
    # Arguments
    dataset = 'ct_scans'
    massif_sample_nums = (0,) # Which MASSIF samples to use with fit and test
    include = 'all' # 'all' or ('TTT-AM-P-1-62',) to use with get_stress_fields
    npz_size = 1000 # Number of volumes per npz container
    num_workers = 12 # Select number of CPU resources
    plot_volumes = False # True or False

    task = 'test' # 'fit', 'test', 'get_stress_fields'
    model_type = '3D' # 2D or 3D
    model_name = 'efficientnet-b0' # '*-b0', '*-b1', '*-b2', '*-b3', '*-b4', '*-b5', '*-b6', or '*-b7'
    encoder_ckpt = None # 'autoencoder/ae_pretraining_ct_scans_all.ckpt' or None
    freeze_encoder = False
    decoder = 'unet' # 'unet' or 'unet++'
    decoder_attention = 'scse' # None, 'se', or 'scse'
    num_channels = 1 # 1 or 3
    num_classes = 2 # int (number of output masks)
    surrogate_ckpt = 'surrogate_model_dir/unet_scse_pretraining_ct_scans_sample_0.ckpt' # 'surrogate_model_dir/unet_scse_pretraining_ct_scans_sample_0.ckpt' or None
    
    # Training Details
    persistent_workers = False
    if task == 'fit':
        max_epochs = 50
        distributed = False
        if distributed: persistent_workers = True
        early_stopping = False
        mixed_precision = True
        
        training_info = {'distributed': distributed, 'max_epochs': max_epochs,
                         'mixed_precision': mixed_precision, 'early_stopping': early_stopping}

    else: training_info = None

    # Hyperparameters
    loss = MSELoss() # MSELoss(), PNormLoss(weight=False, percentile=0.95, p=2, do_root=False), EarthMoversMSELoss(bins=50, p=2, do_root=False, alpha=0.1)
    score = R2Score()
    optimizer_name = 'AdamW'
    batch_size = 6
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
                    'samples_dir': samples_dir, 'sample_nums': massif_sample_nums, 'include': include,
                    'predict': predict, 'npz_size': npz_size, 'plot': plot_volumes,
                    'num_workers': num_workers, 'num_channels': num_channels,
                    'persistent_workers': persistent_workers if persistent_workers else None}
    
    model_info = {'task': task, 'model_type': model_type, 'name': model_name, 'encoder_ckpt': encoder_ckpt,
                  'freeze_encoder': freeze_encoder, 'decoder': decoder, 'decoder_attention': decoder_attention, 
                  'num_channels': num_channels,'num_classes': num_classes, 'surrogate_ckpt': surrogate_ckpt,
                  'training_info': training_info, 'npz_archive': None}
    
    hyperparameters = {'loss': loss, 'score': score, 'optimizer': optimizer_name, 'batch_size': batch_size,
                       'lr': lr, 'sch_factor': sch_factor, 'sch_patience': sch_patience}
    
    # Main
    model = Model(dataset_info, model_info, hyperparameters)
    if task == 'fit': model.fit()
    elif task == 'test': model.test()
    elif task == 'get_stress_fields': model.get_stress_fields()
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
    