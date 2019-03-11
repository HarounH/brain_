from clevr.modules import (
    conv,
    fc,
    fgl_clf,
)


def complex_wedge_fgl_maker(args, loadable_state_dict=None):
    return fgl_clf.WedgeClassifier0(args, loadable_state_dict=loadable_state_dict, complex=True)


versions = {
    'fc': fc.Factored,
    'hugeconv': conv.HugeConv,
    'bigconv': conv.BigConv,
    'bigcc': conv.BigCC,
    'maxbigconv': conv.MaxBigConv,
    'maxbigcc': conv.MaxBigCC,
    'conv': conv.ConvClassifier,
    'cc': conv.CoordConvClassifier,
    'quadfgl': fgl_clf.QuadClassifier0,
    'wedgefgl': fgl_clf.WedgeClassifier0,
    'complexwedgefgl': complex_wedge_fgl_maker,
    'regionfgl': fgl_clf.RegionClassifier0,
}
