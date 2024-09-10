from data_pipeline.DataLoaderLite import NpyDataLoaderLite, TxtDataLoaderLite, JsonDataLoaderLite


classname_map = {
    'npy': NpyDataLoaderLite, 
    'txt': TxtDataLoaderLite, 
    'json': JsonDataLoaderLite
}

class DataLoaderLiteFactory:
    def __init__(self):
        self.valid_classname_list = [
            'NpyDataLoaderLite', 
            'TextDataLoaderLite', 
            'JsonDataLoaderLite'
        ]
        print('FilenameObjFactory built successfully')
    
    def create(self, data_format:str, **kwargs):
        classname = classname_map[data_format]
        return classname(**kwargs)

    def create_v2(self, classname:str, **kwargs):
        assert classname in self.valid_classname_list
        return eval(classname)(**kwargs)
