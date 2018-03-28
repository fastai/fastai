from ..data.custom_dataset_data_loader import CustomDatasetDataLoader

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
