from torch.utils.data import Dataset
from utils import load_info, data_from_file
import cv2, random, numpy as np, json, torch

class AVE(Dataset):
    '''
    Zero-Shot Capable Dataloader for the Audio-Visual Events Dataset. CML only.
    '''
    def __init__(self, rootdir, split, settings, video_transform=None, audio_transform=None, precomputed=True):
        self.settings = json.load(open(settings))
        self.rootdir = rootdir
        self.split = split
        self.precomputed = precomputed
        if precomputed:
            self.spatial_labels = data_from_file(self.rootdir + self.settings['precomputed']['folder'] + self.settings['precomputed']['spatial'])
            self.temporal_labels = data_from_file(self.rootdir + self.settings['precomputed']['folder'] + self.settings['precomputed']['temporal'])
            self.audio_features = data_from_file(self.rootdir + self.settings['precomputed']['folder'] + self.settings['precomputed']['audio'])
            self.video_features = data_from_file(self.rootdir + self.settings['precomputed']['folder'] + self.settings['precomputed']['video'])
        else:
            self.video_transform = video_transform
            self.audio_transform = audio_transform
        self.class_map = json.load(open(self.rootdir + self.settings['classes']))
        if self.split == 'train': # seen training
            if precomputed:
                self.data = data_from_file(self.rootdir + self.settings['precomputed']['folder'] + self.settings['precomputed']['zsl_train'])
            else:
                self.info = load_info(self.rootdir + self.settings['raw']['zsl_train'])
        elif self.split == 'test': # seen testing
            if precomputed:
                self.data = data_from_file(self.rootdir + self.settings['precomputed']['folder'] + self.settings['precomputed']['zsl_test'])
            else:
                self.info = load_info(self.rootdir + self.settings['raw']['zsl_test'])
        elif self.split == 'val': # unseen validation
            if precomputed:
                self.data = data_from_file(self.rootdir + self.settings['precomputed']['folder'] + self.settings['precomputed']['zsl_val'])
            else:
                self.info = load_info(self.rootdir + self.settings['raw']['zsl_val'])

    def __getitem__(self, index):
        if self.precomputed:
            video = torch.from_numpy(self.video_features[self.data[index]]).type(torch.FloatTensor)
            audio = torch.from_numpy(self.audio_features[self.data[index]]).type(torch.FloatTensor)
            temporal_label = self.temporal_labels[self.data[index]]
            spatial_label = self.spatial_labels[self.data[index]]
            class_names = self.get_class_names(spatial_label)
            start, end = 0,10
            for j in range(9,-1,-1):
                if temporal_label[j] == 1:
                    end = j + 1
                    break
            for i in range(10):
                if temporal_label[j] == 1:
                    start = i
                    break
            return video.squeeze(), audio.squeeze(), temporal_label, spatial_label, class_names, start, end
        else:
            video = self.get_video(self.rootdir + self.settings['raw']['video'] + self.info[index][1] + '.mp4')
            audio_file = self.rootdir + self.settings['raw']['audio'] + self.info[index][1] + '.wav' # change to audio at some point
            temporal_label = torch.zeros(self.num_segments)
            temporal_label[list(range(int(self.info[index][3]), int(self.info[index][4])))] = 1 
            spatial_label = torch.zeros(self.num_classes)
            spatial_label[self.class_map[self.info[index][0]]] = 1
            augmented_video = self.video_transform(video)
            # augmented_audio = self.audio_transform(audio)
            return augmented_video, audio_file, spatial_label, temporal_label

    def __len__(self):
        if self.precomputed:
            return len(self.data)
        else:
            return len(self.info)

    def get_video(self, filename):
        """
        (string) filename: .mp4 file
        (FloatTensor) return: uniformly sampled video frame data
        """
        cap = cv2.VideoCapture(filename)
        frames = []
        success, image = cap.read()
        while success:
            frames.append(image)
            success, image = cap.read()
        indicies = sorted(random.sample(range(len(frames)), 160))
        return np.asarray([frames[i] for i in indicies])

    def get_class_names(self, spatial_labels):
        classes = []
        for i in torch.from_numpy(spatial_labels):
            for name, value in self.class_map.items():
                if value == torch.argmax(i, dim=0):
                    classes.append(name)
                    break
        return classes