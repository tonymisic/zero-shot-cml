import h5py, json, math, random, numpy as np, torch

def load_info(filename):
    '''(string) root_dir: root directory of dataset
    (string) filename: .txt file of data annotations
    (list) return: [[class, filename, audio_quality, start, end], ... ]
    '''
    data = []
    f = open(filename, 'r')
    for line in f.readlines():
        line = line.rstrip('\n')
        temp = line.split('&')
        data.append(temp)
    return data

def data_from_file(file):
    '''{string) file: filename of .h5 file
    (ndarray) return: data stored as numpy array
    '''
    with h5py.File(file, 'r') as hf:
        return hf[list(hf.keys())[0]][:]

class GeneralizedZeroShot():
    ''' Spliting AVE into a generalized zero-shot dataset
    '''
    def __init__(self, rootdir, precomputed, settings='settings.json'):
        self.settings = json.load(open(settings))
        self.rootdir = rootdir
        self.precomputed = precomputed
        self.testSplit = self.settings['test_split']
        self.valClasses = self.settings['zsl_classes']
        self.class_map = json.load(open(self.rootdir + self.settings['classes']))
        self.inclusion_list = list(range(len(self.class_map)))
        for i in range(len(self.class_map)):
            if i in self.valClasses:
                self.inclusion_list[i] = 0
            else:
                self.inclusion_list[i] = 1
        
        if self.precomputed:
            self.spatial_labels = data_from_file(self.rootdir + self.settings['precomputed']['folder'] + self.settings['precomputed']['spatial'])
            self.temporal_labels = data_from_file(self.rootdir + self.settings['precomputed']['folder'] + self.settings['precomputed']['temporal'])
            self.trainFile = self.rootdir + self.settings['precomputed']['folder'] + self.settings['precomputed']['zsl_train']
            self.testFile = self.rootdir + self.settings['precomputed']['folder'] + self.settings['precomputed']['zsl_test']
            self.valFile = self.rootdir + self.settings['precomputed']['folder'] + self.settings['precomputed']['zsl_val']
        else:
            self.annotations = self.rootdir + self.settings['annotations']
            self.trainFile = self.rootdir + self.settings['raw']['zsl_train']
            self.testFile = self.rootdir + self.settings['raw']['zsl_test']
            self.valFile = self.rootdir + self.settings['raw']['zsl_val']

        self.indexer, self.counter = {v:0 for _,v in self.class_map.items()}, {v:0 for _,v in self.class_map.items()}
        for i, val in enumerate(self.spatial_labels, 0):
            if self.inclusion_list[np.argmax(val)] == 1:
                self.indexer[np.argmax(val)] += 1
        for index, count in self.indexer.items():
            self.indexer[index] = math.floor(count * self.testSplit)
    
    def split_precomputed(self):
        assert self.precomputed, "precomputed is false, but attempting to split precomputed."
        training, testing, validation = [], [], []
        for i, val in enumerate(self.spatial_labels, 0):
            if self.inclusion_list[np.argmax(val)] == 1:
                if self.indexer[np.argmax(val)] > self.counter[np.argmax(val)]:
                    training.append(i)
                elif np.sum(self.temporal_labels[i]) < 10: # only events that can be localized!
                    testing.append(i)
                self.counter[np.argmax(val)] += 1
            else:
                if np.sum(self.temporal_labels[i]) < 10: # only events that can be localized!
                    validation.append(i)
        hf1, hf2, hf3 = h5py.File(self.trainFile, 'w'), h5py.File(self.testFile, 'w'), h5py.File(self.valFile, 'w')
        hf1.create_dataset('dataset', data=np.array(training))
        hf2.create_dataset('dataset', data=np.array(testing))
        hf3.create_dataset('dataset', data=np.array(validation))
        hf1.close(), hf2.close(), hf3.close()
    
    def split_rawdata(self):
        assert not self.precomputed, "precomputed is true, but attempting to split rawdata."
        
        pass

    def print_classes(self):
        pos_list, neg_list = [], []
        for name, value in self.class_map.items():
            if self.inclusion_list[value] == 1:
                pos_list.append(name)
            else:
                neg_list.append(name)
        print("------------------------------------------------")
        print("Classes in training: " + str(pos_list))
        print("------------------------------------------------")
        print("Classes in validation: " + str(neg_list))
        print("------------------------------------------------")
        print("Split complete!")
