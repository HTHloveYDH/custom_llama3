from data_pipeline.data_loader.PTDataLoaderLite import NpyPTDataLoaderLite, TxtPTDataLoaderLite
from data_pipeline.data_loader.PTDataLoaderLiteV2 import NpyPTDataLoaderLiteV2, TxtPTDataLoaderLiteV2, JsonPTDataLoaderLiteV2
from data_pipeline.data_loader.SFTDataLoaderLite import InstructionSFTDataLoaderLite, DialogSFTDataLoaderLite
from data_pipeline.data_loader.DPODataLoaderLite import BaseDPODataLoaderLite


class DataLoaderLiteFactory:
    classname_map = {
        True: {
            'naive': BaseDPODataLoaderLite
        },
        False: {
            True: {
                'instruction': InstructionSFTDataLoaderLite,
                'dialog': DialogSFTDataLoaderLite
            },
            False: {
                'npy': NpyPTDataLoaderLite,
                'txt': TxtPTDataLoaderLite
            }
        }
    }
    valid_classname_list = [
        'NpyPTDataLoaderLite',
        'TxtPTDataLoaderLite',
        'NpyPTDataLoaderLiteV2',
        'TxtPTDataLoaderLiteV2',
        'JsonPTDataLoaderLiteV2',
        'InstructionSFTDataLoaderLite',
        'DialogSFTDataLoaderLite',
        'BaseDPODataLoaderLite'
    ]

    def __init__(self):
        print('DataLoaderLiteFactory built successfully')

    def create(self, align:bool, dialog:bool, data_format:str, **kwargs):
        assert dialog in [True, False]
        assert data_format in ['npy', 'txt', 'instruction', 'dialog', 'naive']
        if align:
            classname = DataLoaderLiteFactory.classname_map[align][data_format]
            return classname(**kwargs)
        classname = DataLoaderLiteFactory.classname_map[align][dialog][data_format]
        return classname(**kwargs)

    def create_v2(self, classname:str, **kwargs):
        assert classname in DataLoaderLiteFactory.valid_classname_list
        return eval(classname)(**kwargs)
