from data_pipeline.data_loader.PTDataLoaderLite import NpyPTDataLoaderLite, TxtPTDataLoaderLite
from data_pipeline.data_loader.SFTDataLoaderLite import InstructionSFTDataLoaderLite, DialogSFTDataLoaderLite


class DataLoaderLiteFactory:
    classname_map = {
        True: {
            'instruction': InstructionSFTDataLoaderLite, 
            'dialog': DialogSFTDataLoaderLite
        }, 
        False: {
            'npy': NpyPTDataLoaderLite, 
            'txt': TxtPTDataLoaderLite, 
        }
    }
    valid_classname_list = [
        'NpyPTDataLoaderLite', 
        'TxtPTDataLoaderLite', 
        'InstructionSFTDataLoaderLite', 
        'DialogSFTDataLoaderLite'
    ]

    def __init__(self):
        print('DataLoaderLiteFactory built successfully')
    
    def create(self, dialog:bool, data_format:str, **kwargs):
        assert dialog in [True, False]
        assert data_format in ['npy', 'txt', 'instruction', 'dialog']
        classname = DataLoaderLiteFactory.classname_map[dialog][data_format]
        return classname(**kwargs)

    def create_v2(self, classname:str, **kwargs):
        assert classname in DataLoaderLiteFactory.valid_classname_list
        return eval(classname)(**kwargs)
