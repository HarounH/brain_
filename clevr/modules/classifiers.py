from clevr.modules import (
    conv,
    fc,
    fgl_clf,
)


versions = {
    'fc': fc.Factored,
    'conv': conv.ConvClassifier,
    'cc': cc.CoordConvClassifier,
    'fgl_clf': fgl_clf.Classifier0,
}
