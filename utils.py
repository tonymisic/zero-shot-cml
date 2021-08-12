import h5py

def load_info(filename):
    """
    (string) root_dir: root directory of dataset
    (string) filename: .txt file of data annotations
    (list) return: [[class, filename, audio_quality, start, end], ... ]
    """
    data = []
    f = open(filename, 'r')
    for line in f.readlines():
        line = line.rstrip('\n')
        temp = line.split('&')
        data.append(temp)
    return data

def data_from_file(file):
    """
    (string) file: filename of .h5 file
    (ndarray) return: data stored as numpy array
    """
    with h5py.File(file, 'r') as hf:
        return hf[list(hf.keys())[0]][:]

def record_variables():
    return