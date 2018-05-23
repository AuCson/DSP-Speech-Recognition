"""
Author: Xisen Jin

config.py
global config and meta info
"""
import wave
import logging


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
        self.val_instance = 0.2
        self.val_person = 0.1
        self.frame = 0.02
        self.step = 0.01

global_config = Config()
global_meta = Meta()

cfg = global_config
meta = global_meta