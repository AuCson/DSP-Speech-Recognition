"""
Author: Xisen Jin

config.py
global config and meta info
"""
import wave
import logging
import time
import os

class Meta:
    def __init__(self):
        try:
            wav = wave.open('./data/15307130053-02-05.wav', "rb")
            self.num_frame = wav.getnframes()
            self.num_channel = wav.getnchannels()
            self.frame_rate = wav.getframerate()
            wav.close()
        except FileNotFoundError:
            print('Sample file not found. Meta info not loaded')
            return

class Config:
    def __init__(self):
        self.cuda = False
        self.batch_size = 32
        self.data_dir = './data/'
        self.val_instance = 0.0
        self.val_person = 0.2
        self.frame = 0.03
        self.step = 0.01
        self.model_path = 'models/hmrnn-0621.pkl'
        self.monitor_dir = 'test/'
        self.init_logging_handler()
        self.cuda_device = 0
        self.lr = 0.001

        self.use_pitch = False
        self.use_timefeat = False

    def init_logging_handler(self):
        current_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        if not os.path.isdir('./log/'):
            os.makedirs('./log/')
        stderr_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('./log/{}.txt'.format(current_time))
        logger = logging.getLogger('verbose')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(stderr_handler)

        logger = logging.getLogger('mute')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

global_config = Config()
global_meta = Meta()
logger = logging.getLogger('mute')
printer = logging.getLogger('verbose')

cfg = global_config
meta = global_meta
