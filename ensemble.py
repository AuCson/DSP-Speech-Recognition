"""
ensemble model
"""
from model import _ModelBase, RNNModel
from pitch_model import PitchModel
from config import *
from sklearn.metrics import accuracy_score

class EnsembleModel(_ModelBase):
    def __init__(self):
        super().__init__()
        self.mfcc_rnn = RNNModel()
        self.pitch_clf01 = PitchModel([0,1])
        self.pitch_clf67 = PitchModel([6,7])

        self.mfcc_rnn.load()
        self.pitch_clf01.load()
        self.pitch_clf67.load()

        self.mfcc_rnn.clf.eval()

    def test(self):
        dev_data = self.reader.mini_batch_iterator(self.reader.val_person)
        y, pred = [], []
        for itr, total_iter, feat, label, files in dev_data:
            pred_, prob = self.mfcc_rnn.test_iter(itr, total_iter, feat, label, files)
            for i, p in enumerate(pred_):
                if p in [0,1] and prob[i] < 0.8:
                    pred_[i] = self.pitch_clf01.test_iter(*feat[i])
                if p in [6,7] and prob[i] < 0.8:
                    pred_[i] = self.pitch_clf67.test_iter(*feat[i])
            y.extend(label)
            pred.extend(pred_)

            printer.info('%d/%d' % (itr, total_iter))
            # for i,_ in enumerate(pred_):
            #    if pred_[i] != label[i]:
            #       logger.info(files[i])
            # if itr > 1000: break
        acc = accuracy_score(y, pred)
        printer.info(acc)
        # cm = confusion_matrix(y, pred)
        # print(cm)
        return acc

if __name__ == '__main__':
    m = EnsembleModel()
    m.test()