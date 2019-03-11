import torch
from torch import nn
from autoencoder.modules import (
    conv_enc,
    conv_dec,
    fgl_enc,
    fgl_dec,
)


def parse_model_specs(args):
    if args.encoder_type.startswith('fgl'):
        if len(args.encoder_type) > 3:
            raise NotImplementedError('Encoder decoder need different specs... Please implement different classes for desired spec')
            args.encoder_type, args.op_order, args.reduction, args.optimization = args.encoder_type.split('_')
        else:
            args.op_order = "213"  # For encoder. decoder is hard coded with 132 i think.
            args.reduction = "sum"
            args.optimization = "packed1.0"  # Always use tree optimization!
        args.encoder_type = "fgl"
    else:
        args.non_linear = (args.encoder_type[-1] == "_")
        if args.non_linear:
            args.encoder_type = args.encoder_type[:-1]
    return args


encoders = {
    'conv': conv_enc.ConvEncoder0,
    'cc': conv_enc.CoordConvEncoder0,
    'fgl': fgl_enc.FGLEncoder0,
}

decoders = {
    'conv': conv_dec.ConvDecoder0,
    'cc': conv_dec.CoordConvDecoder0,
    'fgl': fgl_dec.FGLDecoder0,
}

masked = {
    'conv': False,
    'cc': False,
    'fgl': True,
}


class AutoEncoder(nn.Module):
    # This allows easier/more efficient use of nn.DataParallel
    def __init__(self,
                 args,
                 loadable_state_dict=None,
                 z_size=128,
                 dropout_rate=0.5
                 ):
        super().__init__()
        self.enc = encoders[args.encoder_type](args, loadable_state_dict=loadable_state_dict)
        self.dec = decoders[args.decoder_type](args, loadable_state_dict=loadable_state_dict)

    def forward(self, x):
        return self.dec(self.enc(x))
