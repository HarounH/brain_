from transfer.modules import (
    fc_clf,
    conv_clf,
    fgl_clf,
)

fgl_optimization_map = {
    "fgl": "tree",
    "rfgl": "packed0.3",
    "randomfgl": "packed0.3",
    "smallfgl": "tree",
    "smallerfgl": "tree",
    "smaller2fgl": "tree",
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
    'smallfgl': fgl_clf.SmallClassifier,
    'smallerfgl': fgl_clf.SmallerClassifier,
    'lin': fc_clf.Linear,
    'cc': conv_clf.CoordConvClassifier0,
    'conv': conv_clf.ConvClassifier0,
}

masked = {
    'smallfgl': True,
    'smallerfgl': True,
    'lin': True,
    'cc': False,
    'conv': False,
}

scheduled = {
    'smallfgl': True,
    'smallerfgl': True,
    'lin': True,
    'cc': True,
    'conv': True,
}
