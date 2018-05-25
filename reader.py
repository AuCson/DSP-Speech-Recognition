import os
import numpy as np
import scipy.io.wavfile as wav
from config import *
import re
import random

class Reader:
    def __init__(self, data_dir='./', debug=False):
        self.data_dir = ''
        self.debug = debug
        if not debug:
            self.train, self.val_person, self.val_inst = self.data_split()
        else:
            self.train = self.val_person = self.val_inst = self.debug_split_from_file()

    def read_one(self, file_name='data/15307130053-15-15.wav'):
        rate, sig = wav.read(file_name)
        return rate, sig

    def debug_split_from_file(self):
        r = open('debug.txt')
        l = []
        for line in r.readlines():
            l.append(line.strip())
        return l

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
                if person not in ['15307130079']:
                    person.add(obj.group(1))
            else:
                print(file)
        person = sorted(list(person))
        inst = sorted(list(inst))
        random.Random(0).shuffle(inst)
        random.Random(0).shuffle(person)
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
        print(len(train_files), len(val_person_files), len(val_inst_files))
        return train_files, val_person_files, val_inst_files

    def read_data(self, files):
        reg = re.compile('(\d+)-(\d+)-(\d+)')
        labels, feat = [],[]
        for file in files:
            label = int(reg.match(file).group(2))
            try:
                rate, sig = wav.read(cfg.data_dir + file)
            except Exception as e:
                print('fail to read %s' % file)
                continue
            sig = sig[:,0] # single channel
            labels.append(label)
            feat.append((sig,rate))
        return feat, labels

    def mini_batch_iterator(self, files, batch_size=None):
        batch_size = cfg.batch_size if batch_size is None else batch_size
        s = 0
        random.Random(0).shuffle(files)
        while s < len(files):
            l = files[s:min(len(files), s+batch_size)]
            s += batch_size
            feat, labels = self.read_data(l)
            yield s, len(files), feat, labels, l

    def iterator(self, files,verbose=True):
        print(files[0], len(files))
        random.Random(0).shuffle(files)
        for s,file in enumerate(files):
            #if verbose:
            #    print(file)

            feat, labels = self.read_data([file])
            if not feat:
                continue
            yield s, 1, feat[0],labels[0], file

