# Models
from efficient_models_pytorch_3d import EfficientUnet3D, EfficientUnetPlusPlus3D

# Install packages
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import MSELoss
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.accelerators import find_usable_cuda_devices
from torchmetrics import R2Score

from baal.active.dataset import ActiveLearningDataset
from baal.active.heuristics import Variance, Random, CombineHeuristics
from baal.utils.pytorch_lightning import ResetCallback
from baal.utils.cuda_utils import to_cuda

import static_frame as sf
from pathlib import Path
import zipfile
from zipfile import ZipFile
import numpy as np
from vedo import Volume, Text2D
from slicer import Slicer3DTwinPlotter
from tqdm import tqdm
import sys
import os
import re
import copy
import itertools
import structlog
from datetime import datetime
import time
log = structlog.get_logger("Active Learning Surrogate Model")


class NPZArchive:
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
    def __init__(self, dataset_info, df):
        # Initialize Parameters
        self.array_labels = df['array_label'].values
        self.input_npzs = df['128_npz'].values
        
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


class SurrogateDataModule(LightningDataModule):
    def __init__(self, dataset_info):
        super().__init__()
        # Initialize Parameters
        df = dataset_info['df'].to_pandas()
        working_dataset_size = dataset_info['working_dataset_size']
        labelled_train_size = dataset_info['labelled_train_size']
        validation_size = dataset_info['validation_size']
        test_size = dataset_info['test_size']

        self.num_workers = dataset_info['num_workers']
        self.persistent_workers = dataset_info['persistent_workers']
        self.batch_size = dataset_info['batch_size']

        df = df.sample(frac=1, random_state=42)

        self._shuffle = True

        # If working_dataset_size is specified, select a subset of the data
        if working_dataset_size is not None:
            _, df = train_test_split(df, test_size=working_dataset_size,
                                     stratify=df['cluster_label'])

        # Split Data
        train_df, self.test_df = train_test_split(df, test_size=test_size,
                                                  stratify=df['cluster_label'])
        
        active_df, self.val_df = train_test_split(train_df, test_size=validation_size,
                                                  stratify=train_df['cluster_label'])
        
        # Get Label Indices for Initial Training Data
        self.active_df = active_df.reset_index(drop=True)
        label_df, _ = train_test_split(self.active_df, train_size=labelled_train_size,
                                       stratify=self.active_df['cluster_label'],
                                       random_state=42)
        label_mask_indices = label_df.index.values

        # Initialize Datasets
        self.active_dataset = ActiveLearningDataset(SurrogateDataset(dataset_info, active_df))
        self.validation_dataset = SurrogateDataset(dataset_info, self.val_df)
        self.test_dataset = SurrogateDataset(dataset_info, self.test_df)

        # Label Initial Training Data
        self.active_dataset.label(label_mask_indices)

    @property
    def has_labelled_data(self):
        return self.active_dataset.n_labelled > 0
    
    def train_dataloader(self):
        return DataLoader(self.active_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers) # persistent_workers=True for DDP

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers) # persistent_workers=True for DDP

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def pool_dataloader(self):
        return DataLoader(self.active_dataset.pool, batch_size=self.batch_size, num_workers=self.num_workers)

    def load_state_dict(self, state_dict):
        self.active_dataset.load_state_dict(state_dict["active_dataset"])
        #self.active_dataset.to(self.device)

    def state_dict(self):
        return {"active_dataset": self.active_dataset.state_dict()}
            

class SurrogateModule(LightningModule):
    def __init__(self, model_info, hyperparameters, datamodule, plot=False):
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
        self.datamodule = datamodule
        self.active_dataset = self.datamodule.active_dataset
    
        self.optimizer_name = hyperparameters['optimizer']
        self.batch_size = hyperparameters['batch_size']
        self.lr = hyperparameters['lr']
        self.sch_factor = hyperparameters['sch_factor']
        self.sch_patience = hyperparameters['sch_patience']

        self.heuristic = hyperparameters['heuristic']
        self.training_epochs = hyperparameters['training_epochs']
        self.query_size = hyperparameters['query_size']
        self.mc_iterations = hyperparameters['mc_iterations']
        self.active_count = 0
        
        self.loss = hyperparameters['loss']
        self.loss_name = self.loss.__class__.__name__

        self.score = hyperparameters['score']
        self.score_name = self.score.__class__.__name__

        self.success_count = 0  # Initialize the success counter
        self.success_threshold = hyperparameters['success_threshold']
        self.validation_size = self.datamodule.val_dataloader().dataset.__len__()
        
    def get_model(self):
        # TODO: Add drop_connect_rate option
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
    
    def step(self, batch, batch_idx, validation=False):
        x, (stress_field, strain_field) = batch
        masks = self.model(x)

        stress_mask = masks[:,0,:,:,:].unsqueeze(1)
        stress_loss = self.loss(stress_mask, stress_field)
        stress_score = self.score(stress_mask.flatten(), stress_field.flatten())

        strain_mask = masks[:,1,:,:,:].unsqueeze(1)
        strain_loss = self.loss(strain_mask, strain_field)
        strain_score = self.score(strain_mask.flatten(), strain_field.flatten())

        loss = stress_loss + strain_loss
        score = (stress_score + strain_score) / 2

        if validation:
            # Check if the maximum predictions are within the threshold of the maximum targets
            for i in range(x.size(0)):  # iterate over samples in the batch
                max_stress_pred = stress_mask[i].max().item()
                max_stress_target = stress_field[i].max().item()
                max_strain_pred = strain_mask[i].max().item()
                max_strain_target = strain_field[i].max().item()

                # Check if the maximum predictions are within the threshold of the maximum targets
                # Initialize a temporary count for this sample
                temp_count = 0
                if abs(max_stress_pred - max_stress_target) / max_stress_target <= self.success_threshold:
                    temp_count += 1
                if abs(max_strain_pred - max_strain_target) / max_strain_target <= self.success_threshold:
                    temp_count += 1
                self.success_count += temp_count / 2

        if self.plot:
            self.plot_fields(stress_field, stress_mask, x)
            self.plot_fields(strain_field, strain_mask, x)

        return loss, score

    def training_step(self, batch, batch_idx):
        if self.datamodule.has_labelled_data:
            self.disable_dropout(self.model)

            loss, score = self.step(batch, batch_idx)

            self.log_dict({f"train_loss({self.loss_name})": loss,
                          f"train_score({self.score_name})": score},
                          batch_size=self.batch_size, on_epoch=True, on_step=False, sync_dist=True)

            return loss

    def validation_step(self, batch, batch_idx):
        loss, score = self.step(batch, batch_idx, validation=True)

        self.log_dict({f"val_loss({self.loss_name})": loss,
                      f"val_score({self.score_name})": score},
                      batch_size=self.batch_size, on_epoch=True, on_step=False, sync_dist=True)
    
    def on_validation_end(self):
        if not self.trainer.sanity_checking:
            if (self.current_epoch + 1) % self.training_epochs != 0:
                return  

            self.active_step()
            self.active_count += 1

            train_size = len(self.datamodule.active_dataset)
            success_score = self.success_count / self.validation_size

            # Log 'train_size' and 'success_score' as separate scalars
            self.logger.experiment.add_scalar('train_size', train_size, self.current_epoch)
            self.logger.experiment.add_scalar('success_score', success_score, self.current_epoch)

            self.success_count = 0

            #self.logger.experiment.add_scalar('uncertainty', uncertainty, self.current_epoch)
           
    def test_step(self, batch, batch_idx):
        loss, score = self.step(batch, batch_idx)
        
        self.log_dict({f"test_loss({self.loss_name})": loss,
                       f"test_score({self.score_name})": score},
                       batch_size=self.batch_size, sync_dist=True)
    
    def predict_step(self, batch, batch_idx):
        # Enable Dropout Layers
        self.enable_dropout(self.model)

        # Get Input Volume
        x, _ = batch

        # Initialize tensors to store the maximum values for each Monte Carlo iteration
        stress_max_values = torch.zeros((x.shape[0], 1, self.mc_iterations), device=x.device)
        strain_max_values = torch.zeros((x.shape[0], 1, self.mc_iterations), device=x.device)

        for i in range(self.mc_iterations):
            # Get the model's predictions
            masks = self(x)

            # Separate the stress and strain fields and add a dimension
            stress_mask = masks[:, 0, :, :, :].unsqueeze(1)
            strain_mask = masks[:, 1, :, :, :].unsqueeze(1)

            # Get the maximum values for each instance in the batch
            max_stress = torch.amax(stress_mask, dim=(2, 3, 4))
            max_strain = torch.amax(strain_mask, dim=(2, 3, 4))

            # Store the maximum values for this iteration in the tensors
            # Tensors should be of shape [batch_size, 1, mc_iterations]
            stress_max_values[:, :, i] = max_stress
            strain_max_values[:, :, i] = max_strain

        return stress_max_values, strain_max_values
    
    def predict_on_dataset_generator(self, dataloader):
        if len(dataloader) == 0:
            return None
        log.info(f"Start Predict: Active Learning Step {self.active_count}", dataset=len(dataloader))
        for idx, batch in enumerate(tqdm(dataloader, total=len(dataloader), file=sys.stdout, desc="Monte Carlo Inference", leave=False)):
            if isinstance(self.trainer.accelerator, CUDAAccelerator):
                batch = to_cuda(batch)
            stress_pred, strain_pred = self.predict_step(batch, idx)
            yield (stress_pred.detach().cpu().numpy(), strain_pred.detach().cpu().numpy())

    def active_step(self):
        pool_dataloader = self.datamodule.pool_dataloader()
        if len(pool_dataloader) > 0:
            predictions_generator = self.predict_on_dataset_generator(dataloader=pool_dataloader)
            if predictions_generator is not None:
                # Create two separate generators from the combined generator
                stress_generator, strain_generator = itertools.tee(predictions_generator, 2)

                # Create stress and strain predictions generators
                stress_predictions = (stress for stress, _ in stress_generator)
                strain_predictions = (strain for _, strain in strain_generator)

                # Combine the stress and strain predictions into a single list of generators
                combined_predictions = [stress_predictions, strain_predictions]

                # Apply the heuristic to the combined predictions
                to_label = self.heuristic(combined_predictions)
                if len(to_label) > 0:
                    self.active_dataset.label(to_label[: self.query_size])
                    # Calculate average uncertainty
                    #average_uncertainty = np.mean(uncertainties)
                    return
        self.should_stop = True
    
    def configure_optimizers(self):
        optimizer = getattr(optim, self.optimizer_name)(self.model.parameters(), lr=self.lr)
        lr_scheduler = {'scheduler': ReduceLROnPlateau(optimizer, mode='min',
                                     factor=self.sch_factor, patience=self.sch_patience),
                        'monitor': f"val_loss({self.loss_name})"}
        return [optimizer], [lr_scheduler]
    
    @staticmethod
    def enable_dropout(model):
        # Eable the dropout layers during during prediction for Monte-Carlo Dropout
        model.train()  # Set the whole model to training mode
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm3d):
                m.eval()  # Set BatchNorm3d layers to evaluation mode

    @staticmethod
    def disable_dropout(model):
        # Disable the dropout layers during training
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout3d):
                m.eval()  # Set Dropout3d layers to evaluation mode

    @staticmethod
    def plot_fields(target, mask, icon):
        # Convert Tensors to Numpy Arrays
        target = target.squeeze().int().cpu().detach().numpy()
        mask = mask.squeeze().int().cpu().detach().numpy()
        icon = icon.squeeze().int().cpu().detach().numpy()

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

        plt = Slicer3DTwinPlotter(vol1, vol2, icon, shape=(1, 2), size = (1800, 950), sharecam=True, bg="white", bg2="lightblue")

        plt.at(0).add(Text2D("Target", s=1.2, pos="top-center"))
        
        plt.at(0).add(Text2D(f"Min = {target_min}, Mean = {target_average:.3f}, Max = {target_max}, SCF = {target_scf:.3f}",
                             s=0.8, bg='grey', alpha=0.5, pos='bottom-right'))
        
        plt.at(1).add(Text2D("Prediction", s=1.2, pos="top-center"))
        
        plt.at(1).add(Text2D(f"Min = {mask_min}, Mean = {mask_average:.3f}, Max = {mask_max}, SCF = {mask_scf:.3f}",
                             s=0.8, bg='grey', alpha=0.5, pos='bottom-right'))
        
        plt.show(viewup='z')
        plt.at(0).reset_camera()
        plt.interactive().close()


# TODO: Figure out how to pull out the actual uncertainties as well to keep track of the average
class Heuristics(CombineHeuristics):
    def __init__(self, heuristics, weights):
        super(Heuristics, self).__init__(heuristics, weights)
    
    def __call__(self, predictions):
        """Rank the predictions according to their uncertainties.

        Return both the rank and associated uncertainties.
        """
        return self.get_ranks(predictions)[0], self.get_ranks(predictions)[1]


class Model:
    def __init__(self, dataset_info, model_info, hyperparameters):
        super(Model, self).__init__()
        # Initialize Parameters
        self.dataset_info = dataset_info
        self.dataset = self.dataset_info['dataset']
        self.dfs_zip = dataset_info['dfs_zip']
        self.samples_dir = dataset_info['samples_dir']
        self.massif_sample = dataset_info['massif_sample']
        self.include = dataset_info['include']

        self.model_info = model_info
        self.surrogate_ckpt = self.model_info['surrogate_ckpt']

        training_info = self.model_info['training_info']
        self.distributed = training_info['distributed']
        self.mixed = training_info['mixed_precision']
        self.early_stopping = training_info['early_stopping']
        self.device = training_info['device']

        self.hyperparameters = hyperparameters
        self.heuristic = self.hyperparameters['heuristic']
        self.active_learning_steps = self.hyperparameters['active_learning_steps']
        self.training_epochs = self.hyperparameters['training_epochs']

        self.loss = self.hyperparameters['loss']
        self.score = self.hyperparameters['score']

        self.setup()

    def setup(self):
        self.model_args = {'model_info': self.model_info, 'hyperparameters': self.hyperparameters}

        self.loss_name = self.loss.__class__.__name__
        self.score_name = self.score.__class__.__name__

        # Setup Devices to Use
        if self.device is None and torch.cuda.is_available():
            self.device = find_usable_cuda_devices()
        else:
            self.device = [self.device] if self.device is not None else None

        # Setup Random Seed and Precision
        seed_everything(42, workers=True)
        torch.set_float32_matmul_precision('medium')

        # Setup Log Directory
        self.log_dir = f"surrogate_model_dir/train_logs/{self.dataset}"
        
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
        
        sample_dir = self.samples_dir / f"{self.massif_sample}"
        file_label = f"{self.dataset}_massif_{self.massif_sample}"
        sample_csv_file = sample_dir / f"{file_label}.csv"

        df = sf.Frame.from_csv(sample_csv_file, encoding='utf8', name=f"{self.massif_sample}")
        df = df.drop[['__index0__']]
        dfs.append(df)

        df = sf.Quilt.from_frames(dfs, retain_labels=True)[self.columns]
        self.dataset_info['df'] = df

    def active_learning(self):
        # Training Pipeline
        datamodule = SurrogateDataModule(self.dataset_info)
        self.model_args['datamodule'] = datamodule

        # TODO: Choose Heuristic based on string ex. 'variance', 'laplace_approximation', 'random'
        heuristic = Variance() if self.heuristic == 'variance' else Random()
        heuristic = CombineHeuristics(heuristics=[heuristic for _ in range(2)], weights=[0.5, 0.5])
        self.model_args['hyperparameters']['heuristic'] = heuristic

        if self.surrogate_ckpt:
            model = SurrogateModule.load_from_checkpoint(self.surrogate_ckpt, **self.model_args,)
        else:
            model = SurrogateModule(**self.model_args)

        self.hyperparameters['heuristic'] = self.heuristic
        version = f"{self.massif_sample}_{self.heuristic}"
        version = self.get_next_version('surrogate_model_dir/train_logs/active_learning', version)

        logger = TensorBoardLogger('surrogate_model_dir/train_logs',
                                   name='active_learning',
                                   version=version,
                                   default_hp_metric=False)

        if self.distributed and torch.cuda.is_available():
            strategy = DDPStrategy(find_unused_parameters=True)
        else:
            strategy = 'auto'

        filename = f"validation_checkpoint_{{epoch:02d}}_{{val_score({self.score_name}):.2f}}"
        validation_checkpoint_callback = ModelCheckpoint(filename=filename,
                                                         save_top_k=1,
                                                         monitor=f"val_score({self.score_name})",
                                                         mode='max')

        #success_checkpoint_callback = ModelCheckpoint(filename='success_checkpoint_{epoch:02d}-{success_count:02d}',
        #                                              save_top_k=1,
        #                                              monitor=f"val_loss({self.loss_name})",
        #                                              mode='max',
        #                                              state_key='success_checkpoint')
        
        reset_checkpoint_callback = ResetCallback(copy.deepcopy(model.state_dict()))
        early_stop_callback = EarlyStopping(monitor=f"val_loss({self.loss_name})", mode='min')
        
        callbacks = [validation_checkpoint_callback, reset_checkpoint_callback]
        if self.early_stopping:
            callbacks.append(early_stop_callback)

        trainer = Trainer(logger=logger,
                          max_epochs=self.active_learning_steps * self.training_epochs,
                          reload_dataloaders_every_n_epochs=1,
                          enable_checkpointing=True,
                          accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                          strategy=strategy,
                          devices=self.device,
                          precision='16-mixed' if self.mixed else 32,
                          callbacks=callbacks)
        
        trainer.fit(model, datamodule=datamodule)
        
        best_model_path = Path(validation_checkpoint_callback.best_model_path)
        best_epoch_search = re.search('epoch=(\d+)', best_model_path.as_posix())
        best_epoch = int(best_epoch_search.group(1))
        
        best_val_score = validation_checkpoint_callback.best_model_score.item()
        best_test_score = trainer.test(ckpt_path='best', datamodule = datamodule)[0][f"test_score({self.score_name})"]
        #best_success_score = success_checkpoint_callback.best_model_score.item()
       
        metrics = {'best_epoch': best_epoch, 'validation_score': best_val_score, 'test_score': best_test_score}#, 'success_score': best_success_score}
        trainer.logger.log_hyperparams(self.hyperparameters, metrics)

    def test(self):
        # Testing Pipeline
        datamodule = SurrogateDataModule(self.dataset_info)
        self.model_args['datamodule'] = datamodule

        if self.surrogate_ckpt:
            model = SurrogateModule.load_from_checkpoint(self.surrogate_ckpt, **self.model_args,)
        else:
            print("No surrogate model checkpoint found. Need checkpoint for testing"); sys.exit()

        logger = TensorBoardLogger(f"surrogate_model_dir/test_logs",
                                   name=f"{self.dataset}",
                                   version=f"{self.massif_sample}_{self.heuristic}",
                                   default_hp_metric=False)

        trainer = Trainer(logger=logger,
                          accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                          devices=1,
                          num_nodes=1)
        
        trainer.test(model, datamodule=datamodule)

    @staticmethod
    def get_next_version(base_dir, base_version):
        # Create the directory if it doesn't exist
        base_dir = Path(base_dir)
        base_dir.mkdir(parents=True, exist_ok=True)

        # Get a list of all existing versions
        existing_versions = os.listdir(base_dir)

        # Extract the numbers from the existing versions that match the base version
        version_numbers = [int(re.search(f'{base_version}_(\d+)', version).group(1)) 
                           for version in existing_versions 
                           if re.match(f'{base_version}_\d+', version)]

        # If no matching versions were found, start at 0
        if not version_numbers:
            return f'{base_version}_0'

        # Otherwise, return the base version followed by the next available number
        return f'{base_version}_{max(version_numbers) + 1}'


def main():
    # TODO: Params loader
    # Arguments
    dataset = 'ct_scans'
    massif_sample = 'stratified_0'  # Which MASSIF sample to use with active_learning and test
    include = 'all'  # 'all' or ('TTT-AM-P-1-62',...) to use with get_stress_fields
    working_dataset_size = None  # None, float (0.0 to 1.0), or int (number of samples)
    num_workers = 12  # Select number of CPU resources for data loading
    plot_volumes = False  # True or False

    task = 'active_learning'  # 'active_learning', 'test'
    model_type = '3D'  # 2D or 3D
    model_name = 'efficientnet-b0'  # '*-b0', '*-b1', '*-b2', '*-b3', '*-b4', '*-b5', '*-b6', or '*-b7'
    encoder_ckpt = None  # None or 'autoencoder/ae_pretraining_ct_scans_all.ckpt'
    freeze_encoder = False  # True or False
    decoder = 'unet'  # 'unet' or 'unet++'
    decoder_attention = 'scse'  # None, 'se', or 'scse'
    decoder_dropout = 0.5  # None or float
    head_dropout = 0.5  # None or float
    num_channels = 1  # 1 or 3
    num_classes = 2  # int (number of output masks)
    surrogate_ckpt = None  # None or 'surrogate_model_dir/unet_scse_pretraining_ct_scans_sample_0.ckpt'
    
    # Training Details
    distributed = False
    persistent_workers = True if distributed else False
    early_stopping = False
    mixed_precision = True
    device = None  # None or select GPU: 0, 1, 2, 3

    # Active Learning Hyperparameters
    heuristic = 'variance'  # 'variance', 'random', TODO: 'laplace_approximation', TODO: multiple weighted heuristics
    labelled_train_size = 750  # Training data size to label initially, float (0.0 to 1.0), or int (number of samples)
    validation_size = 500  # Validation data size, float (0.0 to 1.0), or int (number of samples)
    test_size = 250  # Test data size, float (0.0 to 1.0), or int (number of samples)
    active_learning_steps = 25  # Number of active learning steps to perform for query selection
    training_epochs = 1  # Number of epochs to train the model for each active learning step
    mc_sampling_iterations = 10  # Number of Monte Carlo iterations for uncertainty estimation
    query_size = 40  # Total queries = active_learning_steps * query_size = 25 * 40 = 1000
    
    # Model Hyperparameters
    loss = MSELoss()
    score = R2Score()
    success_threshold = 0.05  # Percentage threshold for success metric
    optimizer_name = 'AdamW'
    batch_size = 6
    lr = 0.0022980
    sch_factor = 0.7
    sch_patience = 0

    # File Setup
    data_dir = Path(f"datasets/{dataset}")
    dfs_zip = data_dir / f"{data_dir.name}_dfs.zip"
    samples_dir = Path(f"surrogate_model_dir/massif_samples/{dataset}")
    
    # Dictionary Setup
    dataset_info = {'dataset': dataset, 'dfs_zip': dfs_zip, 'samples_dir': samples_dir,
                    'massif_sample': massif_sample, 'include': include,
                    'working_dataset_size': working_dataset_size, 'labelled_train_size': labelled_train_size,
                    'validation_size': validation_size, 'test_size': test_size,
                    'num_channels': num_channels, 'batch_size': batch_size, 'num_workers': num_workers, 
                    'persistent_workers': persistent_workers if persistent_workers else None}
    
    training_info = {'distributed': distributed, 'mixed_precision': mixed_precision,
                     'early_stopping': early_stopping, 'device': device}
    
    model_info = {'model_type': model_type, 'name': model_name, 'encoder_ckpt': encoder_ckpt,
                  'surrogate_ckpt': surrogate_ckpt, 'freeze_encoder': freeze_encoder,
                  'decoder': decoder, 'decoder_attention': decoder_attention, 
                  'dropout': {'decoder_dropout': decoder_dropout, 'head_dropout': head_dropout},
                  'num_channels': num_channels,'num_classes': num_classes,
                  'training_info': training_info, 'plot': plot_volumes}
    
    hyperparameters = {'heuristic': heuristic, 'active_learning_steps': active_learning_steps,
                       'training_epochs': training_epochs, 'mc_iterations': mc_sampling_iterations,
                       'query_size': query_size, 'loss': loss, 'score': score,
                       'success_threshold': success_threshold, 'optimizer': optimizer_name,
                       'batch_size': batch_size,'lr': lr, 'sch_factor': sch_factor,
                       'sch_patience': sch_patience}
    
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
    