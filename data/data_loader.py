from .base_dataset import BaseDataset
from data.base_dataset import is_image_file
import os

def CreateDataLoader(opt):
    # from data.custom_dataset_data_loader import CustomDatasetDataLoader
   
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

class CustomDatasetDataLoader():
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        self.opt = opt
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            drop_last=True)

    def load_data(self):
        return self.dataloader

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames)[:int(max_dataset_size)]:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def CreateDataset(opt):
    dataset = AlignedDataset()
    dataset.initialize(opt)
    return dataset

class AlignedDataset(BaseDataset):
    def name(self):
        return 'AlignedDataset'

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))
        self.A_paths = []
        self.B_paths = []
        for AB_path in self.AB_paths:
            # split AB image into A and B
            A_path, B_path = split_AB(AB_path, opt.input_nc, opt.output_nc)
            self.A_paths.append(A_path)
            self.B_paths.append(B_path)

    def __getitem__(self, index):
        A_path = self.A_paths[index % len(self.A_paths)]  # make sure index is within then range
        B_path = self.B_paths[index % len(self.B_paths)]

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # apply image transformation
        A = self.transform(A_img)
        B = self.transform(B_img)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.A_paths)