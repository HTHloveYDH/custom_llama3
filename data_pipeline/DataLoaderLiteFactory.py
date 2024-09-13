from data_pipeline.PTDataLoaderLite import NpyDataLoaderLite, TxtDataLoaderLite, JsonDataLoaderLite
from data_pipeline.SFTDataLoaderLite import SFTDataLoaderLite


class DataLoaderLiteFactory:
    classname_map = {
        True: {'json': SFTDataLoaderLite}, 
        False: {
            'npy': NpyDataLoaderLite, 
            'txt': TxtDataLoaderLite, 
            'json': JsonDataLoaderLite
        }
    }
    valid_classname_list = [
        'NpyDataLoaderLite', 
        'TextDataLoaderLite', 
        'JsonDataLoaderLite', 
        'SFTDataLoaderLite'
    ]

    def __init__(self):
        print('FilenameObjFactory built successfully')
    
    def create(self, dialog:bool, data_format:str, **kwargs):
        assert dialog in [True, False]
        assert data_format in ['npy', 'txt', 'json']
        classname = DataLoaderLiteFactory.classname_map[dialog][data_format]
        return classname(**kwargs)

    def create_v2(self, classname:str, **kwargs):
        assert classname in DataLoaderLiteFactory.valid_classname_list
        return eval(classname)(**kwargs)
