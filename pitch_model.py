"""
Model Based on Pitch - handcrafted features

specifically used for:

00, 01 [yu yin]
06, 07 [bei jing]

"""

from model import _ModelBase
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from features.pitch import *

class PitchModel(_ModelBase):
    def __init__(self, label):
        super().__init__()
        self.label = label

        self.scaler = RobustScaler()
        self.clf = GradientBoostingClassifier()

    def train(self):
        train_itr = self.reader.iterator(self.reader.train)
        X, y = [],[]
        for idx, l, (sig, rate), label, filename in train_itr:
            #reg = re.compile('(\d+)-(\d+)-(\d+).wav')
            #audio = reg.match(filename).group(2)
            #person = reg.match(filename).group(1)
            # if not (person[3:7] == '0713' and int(person[7:]) < 300):
            #    continue
            if label not in self.label:
                continue
            l, r = basic_endpoint_detection(sig, rate)
            sig = preemphasis(sig, coeff=1.0)
            frames = to_frames(sig[l:r], rate)
            feature = pitch_feature(sig[l:r], rate)
            #p.append((feature[0], feature[2]))
            X.append(feature)
            y.append(label)

        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.clf.fit(X, y)

        self.save()

    def test_iter(self, sig, rate):
        l, r = basic_endpoint_detection(sig, rate)
        feature = pitch_feature(sig[l:r], rate)
        X = self.scaler.transform([feature])
        pred = self.clf.predict(X)
        return pred[0]

    def test(self):
        X, y =[],[]
        val_itr = self.reader.iterator(self.reader.val_person)
        for idx, l, (sig, rate), label, filename in val_itr:
            if label not in self.label:
                continue
            l, r = basic_endpoint_detection(sig, rate)
            frames = to_frames(sig[l:r], rate)
            feature = pitch_feature(sig[l:r], rate)
            X.append(feature)
            y.append(label)

        X = self.scaler.transform(X)
        pred = self.clf.predict(X)
        acc = accuracy_score(y, pred)
        print(acc)

    def save(self):
        fs = open('models/scale{}.pkl'.format(''.join([str(_) for _ in self.label])), 'wb')
        fc = open('models/rfc{}.pkl'.format(''.join([str(_) for _ in self.label])), 'wb')
        pickle.dump(self.clf, fc)
        pickle.dump(self.scaler, fs)
        fc.close()
        fs.close()

    def load(self):
        fs = open('models/scale{}.pkl'.format(''.join([str(_) for _ in self.label])), 'rb')
        fc = open('models/rfc{}.pkl'.format(''.join([str(_) for _ in self.label])), 'rb')
        self.scaler = pickle.load(fs)
        self.clf = pickle.load(fc)
        fc.close()
        fs.close()


if __name__ == '__main__':
    m = PitchModel([6,7])
    m.train()
    m.test()
