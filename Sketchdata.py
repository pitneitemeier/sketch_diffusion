from torch.utils.data import Dataset, DataLoader
from itertools import chain
from PIL import Image
from pathlib import Path

class SketchData(Dataset):
    def __init__(self, root_folders, transform, file_format='jpeg'):
        """Constructor
        
        Parameters
        ----------
            root_folders (Sequence of Path/str): list of filepaths to training data './car'
            transform (Compose): A composition of image transforms, see below.
        """
        self.transform = transform
        self._samples = []
        self.file_format = file_format
        for path in root_folders:
            path = Path(path)
            if not (path.exists() and path.is_dir()):
                raise ValueError(f"Data root '{path}' is invalid")
            #store the paths to all images in a list
            self._samples += self._collect_samples(path)
            
    def __getitem__(self, index):
        """Get sample by index
        
        Parameters
        ----------
            index (int)
        
        Returns
        -------
             The indexed sample (Tensor)
        """
        # Access the stored path for the correct index
        path = self._samples[index]
        # Load the image into memory
        img = Image.open(path)
        # Perform transforms, if any.
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __len__(self):
        """Total number of samples"""
        # YOUR CODE HERE
        return len(self._samples)

    def _collect_samples(self, path):
        """Collect all paths
        Helper method for the constructor
        """
        return sorted(list(chain(path.glob('*.%s' % self.file_format))), key=lambda x: x.stem)


class SingleImageData(Dataset):
    def __init__(self, img, epoch_len) -> None:
        '''
        Parameters
        ----------
        img : torch.tensor
            a single input image of shape [c, h, w]
        epoch_len : int
            an arbitrary epoch length, since we have only a single training image. 
            Note that we train on a random step in the diffusion chain for each image in
            a training batch so it makes sense to replicate a single image.
        '''
        super().__init__()
        self.img = img
        self.epoch_len = epoch_len

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        return self.img