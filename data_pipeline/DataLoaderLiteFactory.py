from data_pipeline.data_loader.PTDataLoaderLite import NpyPTDataLoaderLite, TxtPTDataLoaderLite
from data_pipeline.data_loader.PTDataLoaderLiteV2 import NpyPTDataLoaderLiteV2, TxtPTDataLoaderLiteV2, JsonPTDataLoaderLiteV2
from data_pipeline.data_loader.SFTDataLoaderLite import InstructionSFTDataLoaderLite, DialogSFTDataLoaderLite
from data_pipeline.data_loader.DPODataLoaderLite import BaseDPODataLoaderLite


class DataLoaderLiteFactory:
    classname_map = {
        True: {
            'dpo_base': BaseDPODataLoaderLite
        },
        False: {
            True: {
                'sft_instruction': InstructionSFTDataLoaderLite,
                'sft_dialog': DialogSFTDataLoaderLite
            },
            False: {
                'pt_npy': NpyPTDataLoaderLite,
                'pt_txt': TxtPTDataLoaderLite,
                'pt_npy_v2': NpyPTDataLoaderLiteV2,
                'pt_txt_v2': TxtPTDataLoaderLiteV2,
                'pt_json_v2': JsonPTDataLoaderLiteV2
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
        assert data_format in [
            'pt_npy', 'pt_txt', 'pt_npy_v2', 'pt_txt_v2', 'pt_json_v2', 'sft_instruction', 
            'sft_dialog', 'dpo_base'
        ]
        if align:
            classname = DataLoaderLiteFactory.classname_map[align][data_format]
            return classname(**kwargs)
        classname = DataLoaderLiteFactory.classname_map[align][dialog][data_format]
        return classname(**kwargs)

    def create_v2(self, classname:str, **kwargs):
        assert classname in DataLoaderLiteFactory.valid_classname_list
        return eval(classname)(**kwargs)
