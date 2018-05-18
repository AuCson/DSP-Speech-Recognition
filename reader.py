import os
import numpy as np
import scipy.io.wavfile as wav
from config import *
import re


class Reader:
    def __init__(self, data_dir='./'):
        self.data_dir = ''
        self.train, self.val_person, self.val_inst = self.data_split()

    def read_one(self, file_name):
        rate, sig = wav.read(file_name)
        return rate, sig

    def data_split(self):
        """
        init the batch ordering
        :return: 
        """
        l = os.listdir(cfg.data_dir)
        inst, person = set(),set()
        reg = re.compile('(\d+)-(\d+)-(\d+).wav')
        for file in l:
            obj = reg.match(file)
            if obj is not None:
                inst.add(obj.group(3))
                person.add(obj.group(1))
        person = list(person)
        random.shuffle(person)
        inst, person = list(inst), list(person)
        train_person, val_person = person[:-int(cfg.val_person * len(person))], person[-int(cfg.val_person * len(person)):]
        test_inst, val_inst = inst[:-int(cfg.val_instance * len(inst))], inst[-int(cfg.val_instance * len(inst)):]
        train_files, val_person_files, val_inst_files = [],[],[]
        for file in l:
            obj = reg.match(file)
            if obj is not None:
                person, inst = obj.group(1), obj.group(3)
                if person in val_person:
                    val_person_files.append(file)
                elif inst in val_inst:
                    val_inst_files.append(file)
                else:
                    train_files.append(file)
        return train_files, val_person_files, val_inst_files

    def read_data(self, files):
        reg = re.compile('(\d+)-(\d+)-(\d+)')
        labels, feat = [],[]
        for file in files:
            label = int(reg.match(file).group(2))
            try:
                rate, sig = wav.read(cfg.data_dir + file)
            except Exception:
                print('fail to read %s' % file)
                continue
            #sig = sig[:0] # single channel
            labels.append(label)
            feat.append((sig,rate))
        return feat, labels

    def mini_batch_iterator(self, files, batch_size=None):
        batch_size = cfg.batch_size if batch_size is None else batch_size
        s = 0
        random.shuffle(files)
        while s < len(files):
            l = files[s:min(len(files), s+batch_size)]
            s += batch_size
            yield s, len(files), self.read_data(l)

