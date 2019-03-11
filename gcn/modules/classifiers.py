
from gcn.modules import fgl_clf
from gcn.modules import fc_clf
from gcn.modules import conv_clf


fgl_optimization_map = {
    "fgl": "tree",
    "rfgl": "packed0.3",
    "randomfgl": "packed0.3",
    "smallfgl": "tree",
    "smallerfgl": "tree",
    "smaller2fgl": "tree",
    "eqsmallerfgl": "tree",
}

def parse_model_specs(args):
    for fgl_type in fgl_optimization_map.keys():
        if args.classifier_type.startswith(fgl_type):
            if len(args.classifier_type) > len(fgl_type):
                args.classifier_type, args.op_order, args.reduction, args.optimization = args.classifier_type.split('_')
            else:
                args.op_order = "132"
                args.reduction = "sum"
                args.optimization = fgl_optimization_map[fgl_type]  # Always use tree optimization!
            args.classifier_type = fgl_type
            break

    args.non_linear = (args.classifier_type[-1] == "_")
    if args.non_linear:
        args.classifier_type = args.classifier_type[:-1]
    return args


versions = {
    'randomfgl': fgl_clf.RandomFGLClassifier,
    'smallfgl': fgl_clf.SmallClassifier,
    'smallerfgl': fgl_clf.SmallerClassifier,
    'eqsmallerfgl': fgl_clf.EqSmallerClassifier,
    'smaller2fgl': fgl_clf.Smaller2Classifier,
    'fgl': fgl_clf.Classifier,
    'rfgl': fgl_clf.ResidualClassifier,
    # 'fgl0': fgl_clf.Classifier0,
    # 'fgl1': fgl_clf.Classifier1,
    # 'rfgl0': fgl_clf.RClassifier0,
    'fc': fc_clf.Classifier,
    'lin': fc_clf.Linear,
    'redfc': fc_clf.DimensionReduced,
    'cc': conv_clf.CoordConvClassifier0,
    'conv': conv_clf.ConvClassifier0,
    'max': conv_clf.MaxConvClassifier0,
}

masked = {
    'randomfgl': True,
    'smallfgl': True,
    'smallerfgl': True,
    'eqsmallerfgl': True,
    'smaller2fgl': True,
    'fgl': True,
    'rfgl': True,
    # 'fgl0': True,
    # 'fgl1': True,
    # 'rfgl0': True,
    'fc': True,
    'lin': True,
    'redfc': True,
    'cc': False,
    'conv': False,
    'max': False,
}

scheduled = {
    'randomfgl': True,
    'smallfgl': True,
    'smallerfgl': True,
    'eqsmallerfgl': True,
    'smaller2fgl': True,
    'fgl': True,
    'rfgl': True,
    # 'fgl0': True,
    # 'fgl1': True,
    # 'rfgl0': True,
    'fc': True,
    'lin': True,
    'redfc': True,
    'cc': True,
    'conv': True,
    'max': True,
}
