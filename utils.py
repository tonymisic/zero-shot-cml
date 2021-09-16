import h5py, json, math, random, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F
def create_positives(starts):
    labels = torch.zeros([starts.size(0), 10])
    for i in range(starts.size(0)):
        temp = torch.full([10], -1)
        temp[starts[i]] = 1
        labels[i] = temp
    return labels

def create_mask(starts):
    labels = torch.zeros([starts.size(0), 10])
    for i in range(starts.size(0)):
        temp = torch.zeros([10])
        temp[starts[i]] = 1
        labels[i] = temp
    return labels

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
            for _, j in enumerate(val, 0):
                if np.argmax(j) < 28:
                    if self.inclusion_list[np.argmax(j)] == 1:
                        if self.indexer[np.argmax(j)] > self.counter[np.argmax(j)]:
                            training.append(i)
                        else:
                            testing.append(i)
                        self.counter[np.argmax(j)] += 1
                    else:
                        validation.append(i)
                    break
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

class ContrastiveLoss(torch.nn.Module):
    """My contrastive loss for event start prediction
    """
    def __init__(self, temp=0.07, weight=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temp = temp
        self.weight = weight

    def forward(self, video, audio, mask, device):
        assert video.size() == audio.size() and video.size(0) == mask.size(0)
        batch_size = video.size(0)
        batch_loss_a, batch_loss_v = torch.Tensor([0.0]).to(device), torch.Tensor([0.0]).to(device)
        for i in range(batch_size): # i'th sample in batch
            idx = torch.argmax(mask[i]) # index of event start
            pos_sim = torch.exp(F.cosine_similarity(video[i, idx], audio[i, idx], dim=0) / self.temp).to(device)
            v2a_neg_sim, a2v_neg_sim = torch.Tensor([0.0]).to(device), torch.Tensor([0.0]).to(device)
            # Audio Start to Video
            for j in range(video.size(1)): 
                if j != idx:
                    a2v_neg_sim = torch.add(a2v_neg_sim, torch.exp(F.cosine_similarity(video[i, j], audio[i, idx], dim=0) / self.temp))
            batch_loss_a = torch.add(batch_loss_a, -torch.log(pos_sim / a2v_neg_sim))
            # Video Start to Audio
            for j in range(video.size(1)): 
                if j != idx:
                    v2a_neg_sim = torch.add(v2a_neg_sim, torch.exp(F.cosine_similarity(audio[i, j], video[i, idx], dim=0) / self.temp))
            batch_loss_v = torch.add(batch_loss_v, -torch.log(pos_sim / v2a_neg_sim))
        #evenly weighted sum
        return torch.div(torch.add(torch.mul(batch_loss_a, self.weight), torch.mul(batch_loss_v, self.weight)), batch_size) 