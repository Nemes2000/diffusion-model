from data_module.flowers import Flowers102DataModule
from data_module.celeba import CelebADataModule


if __name__ == "__main__":
    data_module = CelebADataModule()
    data_module.prepare_data()
    data_module.setup()  
    print('Data module is succesfully loaded.')
    

