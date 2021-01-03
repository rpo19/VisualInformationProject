from annoy import AnnoyIndex
import os
import configparser

# TODO: valutare se può servire get_nns_by_item per feedback retrieval
class Retriever():

    def __init__(self, indexes_path):
        self.indexes_path = indexes_path
        # create on the first run
        if not os.path.exists(indexes_path):
            os.makedirs(indexes_path)
            with open(os.path.join(indexes_path, 'retrieval_modes.ini'), 'w') as settings: pass

    def create_index(self, df_features, retrieval_mode, metric):
        ### create index
        # get number of features
        num_features = len(df_features.loc[0])
        t = AnnoyIndex(num_features, metric=metric)
        # add features vector to the index
        for index, row in df_features.iterrows():
            t.add_item(index, row.values)    
        # num_trees = num_classes
        num_trees = 9
        t.build(num_trees)
        if t.save(os.path.join(self.indexes_path, retrieval_mode)):
            # save index settings (needed to the future loads)
            self.__save_settings(retrieval_mode, num_features, metric)
        else:
            # failed to save the model
            return False
    
    def delete_index(self, retrieval_mode):
        # delete index file
        index_path = os.path.join(self.indexes_path, retrieval_mode)
        if os.path.exists(index_path):
            os.remove(index_path)
        # delete index configuration from settings
        settings = configparser.RawConfigParser()
        settings.read(os.path.join(self.indexes_path, 'retrieval_modes.ini'))
        if settings.has_section(retrieval_mode):
            settings.remove_section(retrieval_mode)
        # save new settings
        with open(os.path.join(self.indexes_path, 'retrieval_modes.ini'), 'w') as configfile:
            settings.write(configfile)

    # TODO: valutare search_k per tradeoff... se è velocissimo si può alzare per migliorare accuracy
    def retrieve(self, img_features, retrieval_mode, n_neighbours=5, include_distances = False):
        # get settings of the retrieval_mode
        (num_features, metric) = self.__get_settings(retrieval_mode)
        # load index
        t = AnnoyIndex(num_features, metric=metric)
        t.load(os.path.join(self.indexes_path, retrieval_mode))
        # retrieve indexes of the n most similar images
        indexes = t.get_nns_by_vector(img_features, n = n_neighbours, include_distances = include_distances)
        # unload index
        t.unload()
        return indexes


    def __save_settings(self, retrieval_mode, num_features, metric):
        # read from settings file
        settings = configparser.RawConfigParser()
        settings.read(os.path.join(self.indexes_path, 'retrieval_modes.ini'))
        # add new retrieval mode if not present
        if not settings.has_section(retrieval_mode):
            settings.add_section(retrieval_mode)
        # add settings
        settings.set(retrieval_mode,'num_features',num_features)
        settings.set(retrieval_mode,'metric',metric)
        # save new settings
        with open(os.path.join(self.indexes_path, 'retrieval_modes.ini'), 'w') as configfile:
            settings.write(configfile)

    def __get_settings(self, retrieval_mode):
        # read from settings file
        settings = configparser.RawConfigParser()
        settings.read(os.path.join(self.indexes_path, 'retrieval_modes.ini'))
        num_features = int(settings[retrieval_mode]['num_features'])
        metric = settings[retrieval_mode]['metric']
        return (num_features, metric)

if __name__ == "__main__":
    retriever = Retriever('prova')
    print('test')

    

