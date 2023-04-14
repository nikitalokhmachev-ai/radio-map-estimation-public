from insite_map_generator import InsiteMapGenerator
from gudmundson_map_generator import GudmundsonMapGenerator
from map_sampler import MapSampler
import numpy as np
from joblib import Parallel, delayed
import time
import os
import sys
import argparse
import pickle

def get_parser():
    parser = argparse.ArgumentParser(description='Generates train and test set maps. Saves maps as pickled list of three arrays: sampled map, target map, and target mask.')
    parser.add_argument('--num_maps', dest='num_maps', help='Total number of maps to generate.', type=int, default=250000)
    parser.add_argument('--buildings', dest='buildings', help='Include buildings in generated map (opposite command is "--no-buildings")', action='store_true')
    parser.add_argument('--no-buildings', dest='buildings', help='Do not include buildings in generated map (opposite command is "--buildings")', action='store_false')
    parser.add_argument('--batch_size', dest='batch_size', help='Number of maps to save together in a single file / array. \
        This is not necessarily the same as the train or test batch size, which can be set during the loading of the data.', type=int, default=512)
    parser.add_argument('--test_split', dest='test_split', help='Percent of maps to use for testing.', type=float, default=0.1)
    parser.add_argument('--num_cpus', dest='num_cpus', help='Number of CPUs to process maps with.', type=int, default=16)
    return parser

class DatasetGenerator():
    
    # Default values for all of these are stored in parser, but I repeat them here for clarity
    def __init__(self, 
                num_maps=250000,
                buildings=True, 
                batch_size=512, 
                test_split=0.1, 
                num_cpus=16):
        self.num_maps = num_maps
        self.buildings = buildings
        self.batch_size = batch_size
        self.test_split = test_split
        self.num_cpus = num_cpus
        self.time = time.strftime("%m_%d__%Hh_%Mm", time.gmtime())
        # generator and sampler are set in the "generate_train_maps" and "generate_test_maps" methods
        self.generator = None
        self.sampler = None

    def generate_one_map(self, ind_map):
        '''
        This method is called multiple times in parallel from "generate_n_maps".

        Parameters:
            ind_map (int): Dummy variable, used for Parallel processing
        
        Returns:
            data_point (dict):
                'sampled_map' (ndarray): 2 x 32 x 32, first channel is sampled power,
                                        second channel is mask with -1 for buildings, 0 for unsampled locations, 1 for sampled locations
                'target_map' (ndarray): 1 x 32 x 32, power at all locations on map
                'target_mask' (ndarray): 1 x 32 x 32, mask with 0 for buildings, 1 elsewhere
        '''
        if self.generator is None or self.sampler is None:
            raise Exception('You must first call "self.generate_train_maps" or "self.generate_test_maps" to set up appropriate generator and sampler.')

        else:
            # generate full map, building map, and channel power (not used)
            target_map, building_map, channel_power = self.generator.generate()
            sampled_map, sample_mask = self.sampler.sample_map(target_map, building_map)

            # target mask is inverse of building map
            target_mask = 1 - building_map

            # reshaping and adding masks
            sampled_map, target_map, target_mask = self.format_preparation(sampled_map, building_map, sample_mask, target_map, target_mask)

            data_point = {"sampled_map": sampled_map,  # Nf(+1) X Nx X Ny X 
                            "target_map": target_map,  # Nf X Nx X Ny
                            "target_mask": target_mask}  # Nf X Nx X Ny

            return data_point


    def generate_n_maps(self, n_maps, output_dir, file_name):
        '''
        Generate n maps (sampled, target, and mask) and save them to the output directory.

        Parameters
            n_maps (int): Number of maps to generate in a given method call
            output_dir (str): Output directory to save maps
            file_name (str): File name format for maps
        '''
        remainder = n_maps
        batch = 0
        while remainder >= self.batch_size:
            start_time = time.time()
        
            data = Parallel(n_jobs=self.num_cpus, backend='threading')(delayed(self.generate_one_map)(ind_map) 
                                                                            for ind_map in range(int(self.batch_size)))

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'Elapsed time for {self.num_cpus} CPUs to generate and sample {self.batch_size} maps is',
                  time.strftime("%d:%H:%M:%S", time.gmtime(elapsed_time)))

            sampled_maps = np.array([data_point['sampled_map'] for data_point in data])
            target_maps = np.array([data_point['target_map'] for data_point in data])
            target_masks = np.array([data_point['target_mask'] for data_point in data])

            with open(os.path.join(output_dir, f'{file_name}{batch}.pickle'), 'wb') as f:
                pickle.dump([sampled_maps, target_maps, target_masks], f)

            remainder -= self.batch_size
            batch += 1

        if remainder > 0:
            start_time = time.time()

            data = Parallel(n_jobs=self.num_cpus, backend='threading')(delayed(self.generate_one_map)(ind_map) 
                                                                            for ind_map in range(int(remainder)))
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'Elapsed time for {self.num_cpus} CPUs to generate {remainder} maps is',
                    time.strftime("%d:%H:%M:%S", time.gmtime(elapsed_time)))

            sampled_maps = np.array([data_point['sampled_map'] for data_point in data])
            target_maps = np.array([data_point['target_map'] for data_point in data])
            target_masks = np.array([data_point['target_mask'] for data_point in data])

            with open(os.path.join(output_dir, f'{file_name}{batch}.pickle'), 'wb') as f:
                pickle.dump([sampled_maps, target_maps, target_masks], f)


    def generate_train_maps(self):
        '''
        This function sets self.generator and self.sampler to train modes, then calls generate_n_maps.
        If self.buildings=True, then self.generator is InsiteMapGenerator. If False, then self.generator is self.GudmundsonMapGenerator.
        '''
        # Create output directory with timestamp
        output_dir = f'dataset/train_{self.time}'
        file_name = 'train_batch_'
        os.makedirs(output_dir)

        # Initialize Training Generator
        if self.buildings:
            self.generator = InsiteMapGenerator(
                # parameters for InsiteMapGenerator class
                num_tx_per_channel=2,
                l_file_num=np.arange(1,40),
                large_map_size=(244,246),
                filter_map=True,
                filter_size=3,
                inter_grid_points_dist_factor=1,
                # args and kwargs for MapGenerator class
                x_length=100,
                y_length=100,
                n_grid_points_x=32,
                n_grid_points_y=32,
                m_basis_functions=np.array([[1]]),
                noise_power_interval=None)
        
        else:
            self.generator = GudmundsonMapGenerator(
                # parameters for GudmundsonMapGenerator class
                 v_central_frequencies=[1.4e9],
                 tx_power=None,
                 n_tx=2,
                 tx_power_interval=[5, 11], #dBm
                 path_loss_exp=3,
                 corr_shad_sigma2=10,
                 corr_base=0.95,
                 b_shadowing=True,
                 num_precomputed_shadowing_mats=500000,
                 # args and kwargs for MapGenerator class
                 x_length=100,
                 y_length=100,
                 n_grid_points_x=32,
                 n_grid_points_y=32,
                 m_basis_functions=np.array([[1]]),
                 noise_power_interval=None)

        # Initialize Training Sampler
        self.sampler = MapSampler(v_sampling_factor=[0.01, 0.4])

        n_maps = self.num_maps - int(self.num_maps * self.test_split)
        self.generate_n_maps(n_maps, output_dir, file_name)


    def generate_test_maps(self):
        '''
        This function sets self.generator and self.sampler to test modes, then calls generate_n_maps.
        '''
        # Create output directory with timestamp
        output_dir = f'dataset/test_{self.time}'
        os.makedirs(output_dir)
        
        # Initialize Testing Generator
        if self.buildings:
            self.generator = InsiteMapGenerator(
                # parameters for InsiteMapGenerator class
                num_tx_per_channel=2, 
                l_file_num=np.arange(41, 43),
                large_map_size=(244,246),
                filter_map=True,
                filter_size=3,
                inter_grid_points_dist_factor=1,
                # args and kwargs for MapGenerator class
                x_length=100,
                y_length=100,
                n_grid_points_x=32,
                n_grid_points_y=32,
                m_basis_functions=np.array([[1]]),
                noise_power_interval=None)
            
        else:
            self.generator = GudmundsonMapGenerator(
                # parameters for GudmundsonMapGenerator class
                 v_central_frequencies=[1.4e9],
                 tx_power=None,
                 n_tx=2,
                 tx_power_interval=[5, 11], #dBm
                 path_loss_exp=3,
                 corr_shad_sigma2=10,
                 corr_base=0.95,
                 b_shadowing=True,
                 num_precomputed_shadowing_mats=500000,
                 # args and kwargs for MapGenerator class
                 x_length=100,
                 y_length=100,
                 n_grid_points_x=32,
                 n_grid_points_y=32,
                 m_basis_functions=np.array([[1]]),
                 noise_power_interval=None)

        # Testing maps are sampled at set intervals, with an even number of maps per sampling rate
        #sampling_rate = np.concatenate((np.linspace(0.01, 0.2, 10, endpoint=False), np.linspace(0.2, 0.4, 7)), axis=0)
        sampling_rate = np.arange(0.02, 0.42, 0.02)
        n_maps = int(self.num_maps * self.test_split / len(sampling_rate))
        for rate in range(len(sampling_rate)):
            self.sampler = MapSampler(v_sampling_factor=sampling_rate[rate])  # set self.sampler at current sampling rate
            file_name = f'test_{sampling_rate[rate]:.2f}%_batch_'
            self.generate_n_maps(n_maps, output_dir, file_name)


    def format_preparation(self, sampled_map, building_map, sample_mask, target_map, target_mask,):
        """
        This method combines the building_map and sample_mask into a single mask that is concatenated onto sampled_map, 
        then changes the dimensions of sampled_map, target_map, and target_mask so that the channels (Nf) are the first dimension.

        Returns:
        `sampled_map`: Nf(+1) x Nx x Ny array
        `target_map`: Nf x Nx x Ny array
        `target_mask`: Nf x Nx x Ny array

        """
        # put sampled_map and target_map into Nf x Nx X Ny order
        sampled_map = np.transpose(sampled_map, (2,0,1))
        target_map = np.transpose(target_map, (2,0,1))

        # target_mask is expanded to include Nf at the front, and repeated to have the same number of channels Nf as target_map
        target_mask = np.repeat(target_mask[np.newaxis, :, :], target_map.shape[0], axis=0)

        # expand sample_mask and building_map to 1 X Nx X Ny
        sample_mask = np.expand_dims(sample_mask, axis=0)
        building_map = np.expand_dims(building_map, axis=0)

        # combine sample_mask and building_map into a single mask (1 for samples, -1 for buildings, 0 for all else)
        sampled_map = np.concatenate((sampled_map, sample_mask - building_map), axis=0)

        return sampled_map, target_map, target_mask


if __name__ == '__main__':
    parser = get_parser()
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    dataset_generator = DatasetGenerator(
        num_maps=args.num_maps,
        buildings=args.buildings,
        batch_size=args.batch_size,
        test_split=args.test_split,
        num_cpus=args.num_cpus)

    print(f'dataset_{dataset_generator.time}\n')
    print(args, '\n')

    print('Generating test set: \n')
    test_start = time.time()
    dataset_generator.generate_test_maps()
    test_end = time.time()
    duration = test_end - test_start
    print('\nTest set generated in', time.strftime("%H:%M:%S", time.gmtime(duration)), '\n')

    print('Generating train set: \n')
    train_start = time.time()
    dataset_generator.generate_train_maps()
    train_end = time.time()
    duration = train_end - train_start
    print('\nTrain set generated in', time.strftime("%H:%M:%S", time.gmtime(duration)))
