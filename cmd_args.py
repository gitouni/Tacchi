import numpy as np
import yaml


from easydict import EasyDict


def postprocess(args):
    # -------------------- learning --------------------
    if not args.evaluate:
        # -------------------- init --------------------
        if not hasattr(args, 'init'):
            args.init = 'xavier'
        if not hasattr(args, 'gain'):
            args.gain = 1.

        # -------------------- custom lr --------------------
        if hasattr(args, 'custom_lr') and args.custom_lr:
            args.lrs = [float(item) for item in args.lrs.split(',')][::-1]
            args.lr_switch_epochs = [int(item) for item in args.lr_switch_epochs.split(',')][::-1]
            assert (len(args.lrs) == len(args.lr_switch_epochs))

            diffs = [second - first for first, second in zip(args.lr_switch_epochs, args.lr_switch_epochs[1:])]
            assert (np.all(np.array(diffs) < 0))

            args.lr = args.lrs[-1]

    # -------------------- resume --------------------
    if args.evaluate:
        assert hasattr(args, 'resume') or hasattr(args,'pretrained')
        # assert args.resume is not False

    '''
    if not hasattr(args, 'multi_gpu'):
        args.multi_gpu = None

    if not hasattr(args, 'pretrain'):
        args.pretrain = None
    '''

    return args


def parse_args_from_yaml(yaml_path):
    with open(yaml_path, 'r') as fd:
        args = yaml.safe_load(fd)
        args = EasyDict(d=args)
        # args = postprocess(args)
    return args

