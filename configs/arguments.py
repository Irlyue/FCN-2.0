import argparse

ARGUMENTS = []


class Argument:
    parser = None

    def __init__(self, *args, **kwargs):
        if 'default' in kwargs and 'help' in kwargs:
            kwargs['help'] = '{}(default {!r})'.format(kwargs['help'], kwargs['default'])
        self.parser.add_argument(*args, **kwargs)

    @staticmethod
    def set_parser(parser):
        Argument.parser = parser

    @staticmethod
    def get_parser():
        return Argument.parser


def add_arg(func):
    ARGUMENTS.append(func)
    return func


#####################################
#       Customized Arguments        #
#####################################
@add_arg
def lr():
    Argument('--lr', default=1e-3, type=float, help='learning rate')


@add_arg
def lrp():
    Argument('--lrp', default=0.005, type=float,
             help='`lr`*`lrp` will be the learning rate for the backbone network')


@add_arg
def reg():
    Argument('--reg', default=1e-4, type=float,
             help='regularization strength')


@add_arg
def solver():
    Argument('--solver', default='adam', type=str,
             help='solver, choose from(adam, sgd)')


@add_arg
def image_size():
    Argument('-s', '--image_size', default=[224, 224], type=int, nargs=2,
             help='input image size')


@add_arg
def batch_size():
    Argument('-b', '--batch_size', default=8, type=int)


@add_arg
def data():
    Argument('-d', '--data', default='voc2012-train-Segmentation', type=str,
             help='data for training or evaluation')


@add_arg
def model_dir():
    Argument('--model_dir', default='/tmp/fcn', type=str)


@add_arg
def backbone():
    Argument('--backbone', default='resnet_v1_101', type=str,
             help='backbone network for the FCN model')


@add_arg
def n_classes():
    Argument('--n_classes', default=20, type=int,
             help='number of classes, not including the background class')


@add_arg
def ckpt_for_backbone():
    Argument('--ckpt_for_backbone', default=None, type=str,
             help='checkpoint for backbone network')


@add_arg
def n_epochs():
    Argument('--n_epochs', default=1, type=int,
             help='number of epochs for the input function, set it to 1 for evaluation')


@add_arg
def gpu_id():
    Argument('--gpu_id', default=2, type=int,
             help='specify which GPU to use, choose from{0, 1, 2}')


@add_arg
def backbone_stride():
    Argument('--backbone_stride', default=32, type=int,
             help='the output stride out the ResNet backbone network')


def get_default_parser():
    Argument.set_parser(argparse.ArgumentParser())
    for cls in ARGUMENTS:
        cls()
    return Argument.get_parser()


if __name__ == '__main__':
    parser = get_default_parser()
    print(parser.parse_args())
