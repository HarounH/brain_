
# from gcn.modules import fgl_clf
from gcn.modules import fc_clf
from gcn.modules import conv_clf

versions = {
    # 'fgl0': fgl_clf.Classifier0,
    # 'fgl1': fgl_clf.Classifier1,
    # 'rfgl0': fgl_clf.RClassifier0,
    'fc': fc_clf.Classifier,
    'redfc': fc_clf.DimensionReduced,
    'cc': conv_clf.CoordConvClassifier0,
    'conv': conv_clf.ConvClassifier0,
}

masked = {
    'fgl0': True,
    'fgl1': True,
    'rfgl0': True,
    'fc': True,
    'redfc': True,
    'cc': False,
    'conv': False,
}

scheduled = {
    'fgl0': True,
    'fgl1': True,
    'rfgl0': True,
    'fc': False,
    'redfc': False,
    'cc': False,
    'conv': False,
}
