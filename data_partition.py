from torch_dataset import data_partition

'''
PARTITION_INPUT_FILE = './data/medical_csv/data_labels_test.csv' # File with original images and labels
PARTITION_RATIOS = [0.70, 0, 0.3]         # Ratios of training, validation, and test data
PARTITION_UNIFORM = False                     # If to partition data and labels uniformly
PARTITION_OUT_PATH = './data/medical_csv/'       # Out path for .csv files for training, validation, and test data

partition_data(input_file=PARTITION_INPUT_FILE,
               ratios=PARTITION_RATIOS,
               out_path=PARTITION_OUT_PATH,
               uniform=PARTITION_UNIFORM)
'''
data_partition(in_csv_root='./data/medical_data/csv',out_csv_root='./data/medical_data',
               csv_name='test_data.csv',valid_size=0,test_size=0.3)