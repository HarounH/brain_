
from gcn.modules import fgl_clf
from gcn.modules import fc_clf
from gcn.modules import conv_clf

versions = {
    'fgl0': fgl_clf.Classifier0,
    'rfgl0': fgl_clf.RClassifier0,
    'fc': fc_clf.Classifier,
    'cc': conv_clf.CoordConvClassifier0,
    'conv': conv_clf.ConvClassifier0,
}

masked = {
    'fgl0': True,
    'rfgl0': True,
    'fc': True,
    'cc': False,
    'conv': False,
}
