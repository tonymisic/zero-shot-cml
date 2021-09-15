import h5py, json, math, random, numpy as np, torch, torch.nn as nn

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

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss