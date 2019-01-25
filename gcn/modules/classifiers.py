
from gcn.modules import fgl_clf
from gcn.modules import fc_clf
from gcn.modules import conv_clf


def parse_model_specs(args):
    if args.classifier_type.startswith("fgl"):
        if len(args.classifier_type) > 3:
            args.classifier_type, args.op_order, args.reduction, args.optimization = args.classifier_type.split('_')
        else:
            args.op_order = "132"
            args.reduction = "sum"
            args.optimization = "tree"  # Always use tree optimization!
        args.classifier_type = "fgl"
    elif args.classifier_type.startswith("rfgl"):
        if len(args.classifier_type) > 4:
            args.classifier_type, args.op_order, args.reduction, args.optimization = args.classifier_type.split('_')
        else:
            args.op_order = "132"
            args.reduction = "sum"
            args.optimization = "tree"  # Always use tree optimization!
        args.classifier_type = "rfgl"
    else:
        args.non_linear = (args.classifier_type[-1] == "_")
        if args.non_linear:
            args.classifier_type = args.classifier_type[:-1]
    return args


versions = {
    'randomfgl': fgl_clf.RandomFGLClassifier,
    # 'convfgl': fgl_clf.ConvFGLClassifier,
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
}

masked = {
    'randomfgl': True,
    # 'convfgl': True,
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
}

scheduled = {
    'randomfgl': True,
    # 'convfgl': True,
    'fgl': True,
    'rfgl': True,
    # 'fgl0': True,
    # 'fgl1': True,
    # 'rfgl0': True,
    'fc': False,
    'lin': False,
    'redfc': False,
    'cc': True,
    'conv': True,
}
