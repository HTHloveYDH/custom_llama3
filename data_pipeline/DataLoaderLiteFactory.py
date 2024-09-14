from data_pipeline.PTDataLoaderLite import NpyPTDataLoaderLite, TxtPTDataLoaderLite, JsonPTDataLoaderLite
from data_pipeline.SFTDataLoaderLite import SFTDataLoaderLite


class DataLoaderLiteFactory:
    classname_map = {
        True: {'json': SFTDataLoaderLite}, 
        False: {
            'npy': NpyPTDataLoaderLite, 
            'txt': TxtPTDataLoaderLite, 
            'json': JsonPTDataLoaderLite
        }
    }
    valid_classname_list = [
        'NpyPTDataLoaderLite', 
        'TxtPTDataLoaderLite', 
        'JsonPTDataLoaderLite', 
        'SFTDataLoaderLite'
    ]

    def __init__(self):
        print('DataLoaderLiteFactory built successfully')
    
    def create(self, dialog:bool, data_format:str, **kwargs):
        assert dialog in [True, False]
        assert data_format in ['npy', 'txt', 'json']
        classname = DataLoaderLiteFactory.classname_map[dialog][data_format]
        return classname(**kwargs)

    def create_v2(self, classname:str, **kwargs):
        assert classname in DataLoaderLiteFactory.valid_classname_list
        return eval(classname)(**kwargs)
