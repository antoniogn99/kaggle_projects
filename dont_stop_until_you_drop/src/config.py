import os


directory = 'C:\\Users\\anton\\kaggle\\kaggle_projects\\dont_stop_until_you_drop\\'
data_path = os.path.join(directory, 'data')
train_csv_path = os.path.join(data_path, 'train.csv')
train_images_path = os.path.join(data_path, 'images', 'train_images')
test_images_path = os.path.join(data_path, 'images', 'test_images')
NUM_FOLDS = 10
models_directory = os.path.join(directory, 'models')