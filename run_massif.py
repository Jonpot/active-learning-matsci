# Import Packages
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import static_frame as sf
import pandas as pd
from pathlib import Path
import zipfile
from zipfile import ZipFile
import numpy as np
from vedo import Volume, show
from slicer import Slicer3DPlotter
import h5py
from subprocess import Popen, PIPE, STDOUT
import fnmatch
import shutil
from tqdm import tqdm
import time
import math


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

        with self.zipf.open(filename, 'w', force_zip64=True) as vol_file:
            self.write_array(vol_file, vol,
                             allow_pickle=self.allow_pickle,
                             pickle_kwargs=self.pickle_kwargs)

    def close(self):
        # Close NPZ File
        self.zipf.close()


class PoreDataset(Dataset):
    def __init__(self, df):
        # Initialize Parameters
        self.array_labels = df['array_label'].values
        self.input_npzs = df['128_npz'].values

    def __getitem__(self, index):
        # Get Input Volume
        with ZipFile(self.input_npzs[index], 'r') as npz_file:
            np_vol = np.load(npz_file.open(f"{self.array_labels[index]}.npy"))

        # Return Volume and Label
        return np_vol, self.array_labels[index]

    def __len__(self):
        # Get Length of Dataset
        return len(self.array_labels)
    
    @staticmethod
    def plot_volume(np_vol):
        # Plot Volume
        vol =  Volume(np_vol)
        show(vol, axes=9, interactive=True, viewup='z', new=True)


class FeatureDataset(Dataset):
    def __init__(self, df):
        # Initialize Parameters
        self.array_labels = df['array_label'].values
        self.input_npzs = df['feature_npz'].values

    def __getitem__(self, index):
        # Get Input Volume
        with ZipFile(self.input_npzs[index], 'r') as npz_file:
            np_vol = np.load(npz_file.open(f"{self.array_labels[index]}.npy"))

        # Return Volume and Label
        return np_vol, self.array_labels[index]

    def __len__(self):
        # Get Length of Dataset
        return len(self.array_labels)
    
    @staticmethod
    def plot_volume(np_vol):
        # Plot Volume
        vol =  Volume(np_vol)
        show(vol, axes=9, interactive=True, viewup='z', new=True)


class FieldDataset(Dataset):
    def __init__(self, df):
        # Initialize Parameters
        self.array_labels = df['array_label'].values
        self.fields_npzs = df['fields_npz'].values
        self.pore_normal_areas = df['pore_normal_area'].values

    def __getitem__(self, index):
        # Get Input Volume
        with ZipFile(self.fields_npzs[index], 'r') as npz_file:
            fields_array = np.load(npz_file.open(f"{self.array_labels[index]}.npy"))

        # Return Volume and Label
        return fields_array, self.array_labels[index], self.pore_normal_areas[index]

    def __len__(self):
        # Get Length of Dataset
        return len(self.array_labels)
    
    @staticmethod
    def plot_volume(np_vol):
        # Plot Volume
        vol =  Volume(np_vol)
        show(vol, axes=9, interactive=True, viewup='z', new=True)


class MASSIF():
    def __init__(self, dataset_info, sample_info, simulation_info):
        # Initialize Parameters
        self.dataset_info = dataset_info
        self.dataset = self.dataset_info['dataset']
        self.samples_dir = self.dataset_info['samples_dir']
        self.dfs_zip = self.dataset_info['dfs_zip']
        self.include = dataset_info['include']
        self.npz_size = dataset_info['npz_size']
        self.plot = dataset_info['plot']

        self.num_samples = sample_info['num_samples']
        self.sample_size = sample_info['sample_size']
        self.stratify_label = sample_info['stratify_label']

        self.sample_nums = simulation_info['sample_nums']
        self.strain_fields = simulation_info['strain_fields']
        self.el_strain_fields = simulation_info['el_strain_fields']
        self.stress_fields = simulation_info['stress_fields']
        self.euler_angles = simulation_info['euler']
        self.dim = simulation_info['dim']

        self.np = simulation_info['np']
        self.eqincr = simulation_info['eqincr']
        self.nsteps = simulation_info['nsteps']
        self.itmax = simulation_info['itmax']
        self.error = simulation_info['error']
        
        self.input_file = 'tmp/pore.dream3d'
        self.count = 0

        self.setup()

    def setup(self):
        # Load Bus
        self.bus = sf.Bus.from_zip_npz(self.dfs_zip)
        self.ct_scans = list(self.bus.keys())
        
        # Select CT Scans
        if self.include != 'all':
            scan_indexes = [i for i in range(len(self.ct_scans)) if self.ct_scans[i] in self.include]
            self.bus = self.bus.iloc[scan_indexes]
            self.ct_scans = list(self.bus.keys())


    def get_sample_dataset(self):
        # Load Dataframe
        print("Making MASSIF Training Sample")
        df = sf.Quilt(self.bus, retain_labels=True).to_frame().to_pandas()
        if self.stratify_label == 'cluster_label':
            df = df.loc[df['cluster_label'] != -1]

        #unique_counts = df['cluster_label'].value_counts()
        #print(unique_counts)

        for sample_num in range(self.num_samples):
            # Setup File Info
            sample_dir = self.samples_dir / f"sample_{sample_num}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{self.dataset}_massif_sample_{sample_num}.csv"
            sample_csv_file = sample_dir / filename

            # Separate the large cluster from the rest of the data
            large_cluster = df[df['cluster_label'] == 18]
            other_clusters = df[df['cluster_label'] != 18]

            # Sample 30% of the large cluster
            large_cluster_sample = large_cluster.sample(n=int(0.3*self.sample_size), random_state=42)

            # Calculate the remaining sample size
            remaining_sample_size = self.sample_size - len(large_cluster_sample)

            # Stratified sampling based on the 'cluster' column
            other_clusters_sample, _ = train_test_split(other_clusters, train_size=remaining_sample_size,
                                                        stratify=other_clusters[self.stratify_label], random_state=42)

            # Combine the samples from the large cluster and the other clusters
            sample_df = pd.concat([large_cluster_sample, other_clusters_sample])

            sample = sf.Frame.from_pandas(sample_df, name=f"sample_{sample_num}").sort_index()
            sample = sample.relabel(range(len(sample)))

            # Calculate the proportions of each cluster
            clusters = sample_df['cluster_label'].values
            cluster_labels, cluster_counts = np.unique(clusters, return_counts=True)
            cluster_proportions = cluster_counts / cluster_counts.sum()

            # Print out the cluster labels, counts, and proportions
            for label, count, proportion in zip(cluster_labels, cluster_counts, cluster_proportions):
                print(f"Cluster {label}: {count} instances, {100*proportion:.2f}%")

            sample.to_csv(sample_csv_file)


    def run(self):
        # EulerAngles
        eul = np.ones((self.dim,self.dim,self.dim))*self.euler_angles[0]
        eua = np.ones((self.dim,self.dim,self.dim))*self.euler_angles[1]
        eub = np.ones((self.dim,self.dim,self.dim))*self.euler_angles[2]
        euler_matrix = np.vstack([eul,eua,eub])
        self.euler_matrix = np.transpose(euler_matrix).reshape((self.dim,self.dim,self.dim,3)).astype('f4')

        for sample_num in self.sample_nums:
            # Load Dataframe
            sample_dir = self.samples_dir / f"sample_{sample_num}"
            filename = f"{self.dataset}_massif_sample_{sample_num}.csv"
            sample_csv_file = sample_dir / filename

            fields_dir = sample_dir / 'all_fields'
            fields_dir.mkdir(parents=True, exist_ok=True)
            
            df = sf.Frame.from_csv(sample_csv_file, encoding='utf8', name=f"{self.dataset}_massif_sample_{sample_num}")
            df = df.drop[['__index0__']]
        
            num_loops = math.ceil(len(df)/self.npz_size)
            fields_npzs = []
            for loop_num in range(num_loops):
                start_index = loop_num * self.npz_size
                stop_index = start_index + self.npz_size if loop_num+1 != num_loops else None
                
                df_slice = df.iloc[start_index:stop_index]
                pore_dataset = PoreDataset(df_slice[['array_label', '128_npz']])

                # Open NPZ Archive
                fields_npz = fields_dir / f"{self.dataset}_massif_sample_{sample_num}_{loop_num}.npz"
                npz_file = NPZArchive(fields_npz, compress=True)

                for index in range(len(pore_dataset)):
                    start_time = time.time()
                    fields_npzs.append(fields_npz.as_posix())

                    # Create Temporary Directory to Work In
                    self.tmp_path = Path('tmp')
                    if self.tmp_path.exists():
                        shutil.rmtree(self.tmp_path)
                    self.tmp_path.mkdir()

                    # Get Volume and Label for MASSIF
                    np_vol, label = pore_dataset.__getitem__(index)

                    # Buffer Volume
                    np_vol = np_vol[4:-4,4:-4,:]
                    np_vol = np.pad(np_vol, ((4,4),(4,4),(0,0)), constant_values=1)

                    # Microstructure
                    feature_id = np.expand_dims(np.transpose(np.copy(np_vol)), -1)
            
                    phase = np.copy(np_vol)
                    phase[phase > 0] = 2 # void phase 2
                    phase[phase == 0] = 1 # solid phase 1
                    phase = np.expand_dims(np.transpose(phase), -1)

                    # Create input file
                    new_data = h5py.File(self.input_file,'w')
                    DataContainers = new_data.create_group('DataContainers')
                    EulerAngles = DataContainers.create_dataset('EulerAngles', data=self.euler_matrix, dtype = 'f4')                
                    FeatureIds = DataContainers.create_dataset('FeatureIds', data=feature_id, dtype='i4')
                    Phases = DataContainers.create_dataset('Phases', data=phase, dtype='i4')
                    new_data.close()

                    # Update params in options.in
                    output_prefix = f"tmp/{label}_"
                    dimensions = f"{self.dim} {self.dim} {self.dim} {self.dim*self.dim*self.dim}"
                    params = (output_prefix, self.input_file, dimensions, self.eqincr, self.nsteps, self.error, self.itmax)
                    patterns = ('--output_prefix*', '--hdfmicrostructure_file*', '--dimensions*',
                                '--eqincr*', '--nsteps*', '--error*', '--itmax*')
                    self.update_params(params, patterns)

                    # Run MASSIF
                    cli_command = ['mpirun','--oversubscribe', '-np', str(self.np), 'pevpmpifft-hdf']
                    process = Popen(cli_command, stdout=PIPE, stderr=STDOUT, text=True)
                    #cli_command = ['nohup', 'mpirun','--oversubscribe', '-np', str(self.np), 'pevpmpifft-hdf']
                    #process = Popen(cli_command, stdout=PIPE, stderr=STDOUT, text=True, preexec_fn=os.setpgrp)
                    self.print_stdout(process)

                    # Extract Stress Field Data and Write Fields to File
                    fields = self.get_field_data()

                    if self.plot:
                        self.plot_volume(fields[14], field=True, icon=np_vol)

                    npz_file.write(fields, label)

                    # Process count
                    process_time = round(time.time() - start_time, 2)
                    print(f"\nSubprocess {self.count} complete in {process_time}s\n")
                    self.count += 1

                # Close NPZ Archive
                npz_file.close()

            # Add Stress Field Data to Dataframe
            if 'fields_npz' in df.keys():
                df = df.drop[['fields_npz', 'mean_stress_field_npz', 'mean_strain_field_npz', 'sscf',
                              'min_stress', 'max_stress', 'average_stress', 'stress_cf',
                              'min_strain', 'max_strain', 'average_strain', 'strain_cf']]
                
            fields_npzs = sf.Series(fields_npzs, name='fields_npz')
            df = df.insert_before('ct_scan', fields_npzs)

            field_dataset = FieldDataset(df[['array_label', 'fields_npz', 'pore_normal_area']])
            mean_fields_npzs, mean_fields_data = self.extract_mean_field_data(field_dataset)

            df = df.insert_after('fields_npz', mean_fields_npzs)
            df = df.insert_after(df.columns[-1], mean_fields_data)
            df.to_csv(sample_csv_file)
            
            shutil.rmtree(self.tmp_path)

    def get_field_data(self):
        # TODO: Incorporate Fatigue and Failure Analysis
        # Get Stress Field Data
        h5_path = list(self.tmp_path.glob('*data.h5'))[0]
        h5_file = h5py.File(h5_path, 'r')
        step = list(h5_file.get('3Ddatacontainer').keys())[-2]
        
        fields = []
        for strain_field in self.strain_fields:
            field = np.transpose(h5_file.get(f"3Ddatacontainer/{step}/Datapoint/Dfields/{strain_field}")[:])
            field = field.astype(np.float16) # TODO: Compare float16 size to float32 size
            fields.append(field)

        for el_strain_field in self.el_strain_fields:
            field = np.transpose(h5_file.get(f"3Ddatacontainer/{step}/Datapoint/Elastic strain/{el_strain_field}")[:])
            field = field.astype(np.float16)
            fields.append(field)

        for stress_field in self.stress_fields:
            field = np.transpose(h5_file.get(f"3Ddatacontainer/{step}/Datapoint/Sfields/{stress_field}")[:])
            field = field.astype(np.float16)
            fields.append(field)

        fields = np.stack(fields, axis=0)

        return fields
    
    def extract_mean_field_data(self, field_dataset):
        sample_dir = self.samples_dir / 'sample_0'

        stress_fields_dir = sample_dir / 'mean_stress_fields'
        stress_fields_dir.mkdir(exist_ok=True)

        strain_fields_dir = sample_dir / 'mean_strain_fields'
        strain_fields_dir.mkdir(exist_ok=True)

        # Open NPZ Archive
        stress_fields_npz = stress_fields_dir / f"{self.dataset}_massif_sample_0_mean_stress_field.npz"
        stress_fields_file = NPZArchive(stress_fields_npz, compress=True)

        strain_fields_npz = strain_fields_dir / f"{self.dataset}_massif_sample_0_mean_strain_field.npz"
        strain_fields_file = NPZArchive(strain_fields_npz, compress=True)

        min_stresses = []
        max_stresses = []
        average_stresses = []
        stress_cfs = []
        min_strains = []
        max_strains = []
        average_strains = []
        strain_cfs = []
        sscfs = []
        for index in tqdm(range(len(field_dataset)), desc='Extracting Mean Field Data'):
            fields_array, label, pore_normal_area = field_dataset.__getitem__(index)

            # Stress Field Data
            mean_stress_field = fields_array[14].astype(np.float32)
            min_stress = np.min(mean_stress_field)
            max_stress = np.max(mean_stress_field)
            average_stress = np.average(mean_stress_field)
            stress_cf = max_stress / average_stress

            min_stresses.append(min_stress)
            max_stresses.append(max_stress)
            average_stresses.append(average_stress)
            stress_cfs.append(stress_cf)

            mean_stress_field = mean_stress_field.astype(np.float16)
            stress_fields_file.write(mean_stress_field, label)

            # Strain Field Data
            D11_strain_field = fields_array[0]
            D22_strain_field = fields_array[3]
            D33_strain_field = fields_array[5]

            mean_strain_field = ((D11_strain_field + D22_strain_field + D33_strain_field) / 3).astype(np.float32)
            min_strain = np.min(mean_strain_field)
            max_strain = np.max(mean_strain_field)
            average_strain = np.average(mean_strain_field)
            strain_cf = max_strain / average_strain

            min_strains.append(min_strain)
            max_strains.append(max_strain)
            average_strains.append(average_strain)
            strain_cfs.append(strain_cf)

            mean_strain_field = mean_strain_field.astype(np.float16)
            strain_fields_file.write(mean_strain_field, label)

            # Stress-Strain Concentration Factor
            sscf = math.sqrt(stress_cf * strain_cf * math.sqrt(pore_normal_area))
            sscfs.append(sscf)
            breakpoint

        stress_fields_file.close()
        strain_fields_file.close()

        stress_fields_npzs = sf.Series([stress_fields_npz]*len(field_dataset), name='mean_stress_field_npz')
        strain_fields_npzs = sf.Series([strain_fields_npz]*len(field_dataset), name='mean_strain_field_npz')
        mean_fields_npzs = sf.Frame.from_concat((stress_fields_npzs, strain_fields_npzs), axis=1)

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
        
        return mean_fields_npzs, mean_fields_data

    @staticmethod
    def update_params(params, patterns):
        with open('options.in', 'r+') as options_file:
            options = options_file.readlines()
            for param, pattern in zip(params, patterns):
                option = pattern[:-1]
                match = fnmatch.filter(options, pattern)
                index = options.index(match[0])
                options[index] = f"{option} {param}\n"
            
            options_file.seek(0)
            options_file.writelines(options)
    
    @staticmethod
    def print_stdout(process):
        while process.poll() is None:
            output = process.stdout.readline()
            print(output.strip())

    @staticmethod
    def plot_volume(np_vol, field=False, icon=None, interactive=True):
        # Create vedo Volume
        np_vol = np_vol.astype('f4')
        vol =  Volume(np_vol)
        
        # Plot Stress Field
        if field == True:
            min = np.min(np_vol)
            max = np.max(np_vol)
            average = np.average(np_vol)
            scf = max / average

            print("######################################")
            print("Prediction")
            print(f"Min = {min}")
            print(f"Max = {max}")
            print(f"Average = {average}")
            print(f"SCF = {scf}")
            print("######################################\n")

            #icon = Volume(icon[20:-20,20:-20,20:-20])
            icon = Volume(icon)
        
            plt = Slicer3DPlotter(
            vol,
            icon=icon,
            icon_size=0.15,
            bg="white",
            bg2="lightblue",
            #cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
            cmaps=('jet',),
            use_slider3d=False)

        # Plot Volume
        else:
            show(vol, axes=1, interactive=interactive, viewup='z', new=True)


def main():
    # TODO: Increment sample number based on current latest sample num
    """Changeable parameters"""
    # Arguments
    dataset = 'ct_scans'
    include = 'all' # 'all' or ('TTT-AM-P-1-62',)
    task = 'sample' # 'sample', 'run'
    npz_size = 100 # Number of volumes per npz container
    plot_volumes = False # 'True' or 'False'

    # Sampling Parameters
    num_samples = 1 # Number of samples to generate
    sample_size = 2500 # Number of volumes per sample
    stratify_label = None # Data column used for stratification: None or 'ct_scan', 'cluster_label', etc.
    
    # Simulation Parameters
    sample_nums = (0,) # Which samples to run MASSIF on
    strain_fields = ('D11', 'D12', 'D13', 'D22', 'D23', 'D33', 'EVM') # To save
    elastic_strain_fields = ('Dilation', 'El11', 'El12', 'El13', 'El22', 'El23', 'El33')
    stress_fields = ('Mean_Stress', 'S11', 'S12', 'S13', 'S22', 'S23', 'S33', 'SVM')
    euler_angles = (0,0,0)  # Euler angles for euler_matrix generation
    dim = 128
         
    num_processes = 32  # Number of cores to use for mpirun
    strain_increment = 0.001  # Hyperparameter
    num_steps = 1  # Hyperparameter
    max_iterations = 100  # Hyperparameter
    error = 0.000001  # Hyperparameter

    # Fields Stack Indices
    fields = {strain_fields: {'D11': 0, 'D12': 1, 'D13': 2, 'D22': 3, 'D23': 4, 'D33': 5, 'EVM': 6},
              elastic_strain_fields: {'Dilation': 7, 'El11': 8, 'El12': 9, 'El13': 10, 'El22': 11, 'El23': 12, 'El33': 13},
              stress_fields: {'Mean_Stress': 14, 'S11': 15, 'S12': 16, 'S13': 17, 'S22': 18, 'S23': 19, 'S33': 20, 'SVM': 21}}
    
    # File Setup
    samples_dir = Path(f"surrogate_model_dir/massif_samples/{dataset}")
    samples_dir.mkdir(parents=True, exist_ok=True)
    dfs_zip = Path(f"datasets/{dataset}/{dataset}_dfs.zip")

    # Dictionary Setup
    dataset_info = {'dataset': dataset, 'samples_dir': samples_dir, 'dfs_zip': dfs_zip,
                    'include': include, 'plot': plot_volumes, 'npz_size': npz_size, 'npz_archive': False}
    
    sample_info = {'num_samples': num_samples, 'sample_size': sample_size, 'stratify_label': stratify_label}

    simulation_info = {'sample_nums': sample_nums, 'strain_fields': strain_fields,
                       'el_strain_fields': elastic_strain_fields, 'stress_fields': stress_fields,
                       'euler': euler_angles, 'dim': dim, 'np': num_processes,'eqincr': strain_increment,
                       'nsteps': num_steps, 'itmax': max_iterations, 'error': error}

    # Main
    simulation = MASSIF(dataset_info, sample_info, simulation_info)
    if task == 'sample': simulation.get_sample_dataset()
    elif task == 'run': simulation.run()

if __name__ == "__main__":
    start_time = time.time()
    
    main()
    
    if round(((time.time() - start_time))/60, 2) >= 60:
        print(f'\nFinished in {round(((time.time() - start_time))/3600, 2)} hours')
    else: print(f'\nFinished in {round(((time.time() - start_time))/60, 2)} minutes')
