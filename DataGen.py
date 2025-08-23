from tensorflow.keras.utils import Sequence
import tensorflow as tf

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, list_IDs,to_fit=True, batch_size=32, dim=(8 ,512, 512), shuffle=True):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.paths = paths
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        # Generate data

        if self.to_fit:
            X,y = self._generate_Xy(list_IDs_temp)
            return X, y
        else:
            X = self._generate_X(list_IDs_temp)
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        
        
        
    def _generate_Xy(self, list_IDs_temp):
        """Generates data containing batch_size images
        :param list_IDs_temp: list of label ids to load
        :return: batch of images
        """
        # Initialization
        X1 = np.zeros((self.batch_size,256,256,1))
        X2 = np.zeros((self.batch_size,256,256,1))
        X3 = np.zeros((self.batch_size,256,256,1))
        X4 = np.zeros((self.batch_size,256,256,1))
        Y = np.zeros((self.batch_size,256,256,3))
        # Generate data
        global Images
        global Labels
        for i, ID in enumerate(list_IDs_temp):
            IMG = np.load(paths[ID])
            
            X1[i,:,:,0] = IMG[:,:,0]
            X2[i,:,:,0] = IMG[:,:,1]
            X3[i,:,:,0] = IMG[:,:,2]
            X3[i,:,:,0] = IMG[:,:,3]
            Y[i,:,:,0] = IMG[:,:,4]
#            
        return [X1,X2,X3,X4],Y