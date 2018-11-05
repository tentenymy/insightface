from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import logging
import argparse
from datetime import datetime

import mxnet as mx
import mxnet.optimizer as optimizer

from image_iter import FaceImageIter
from data import FaceImageIterList
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
import face_image
sys.path.append(os.path.join(os.path.dirname(__file__), 'eval'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'symbols'))
import fresnet
import finception_resnet_v2
import fmobilenet
import fmobilenetv2
import fmobilefacenet
import fxception
import fdensenet
import fdpn
import fnasnet
import spherenet
import fmnasnet
import verification_mxnet_meiyi

from mxboard import SummaryWriter


logger = logging.getLogger()
logger.setLevel(logging.INFO)

args = None


# set logging configuration
def set_log_config(log_path, file_level=logging.INFO, console_level=logging.INFO):
    with open(log_path, 'a+') as file:
        print("write log")
        file.write('\n')

    logger.setLevel(file_level)

    log_formatter_for_file = logging.Formatter("%(asctime)s\t%(message)s", "%Y-%m-%d %H:%M")
    logfile = logging.FileHandler(log_path)
    logfile.setLevel(file_level)
    logfile.setFormatter(log_formatter_for_file)
    logger.addHandler(logfile)

    log_formatter_for_console = logging.Formatter('%(message)s')
    logconsole = logging.StreamHandler()
    logconsole.setLevel(console_level)
    logconsole.setFormatter(log_formatter_for_console)
    logger.addHandler(logconsole)


def get_config(config_path, no_batch, args):
    # check parameter setting
    with open(config_path, 'r') as config_file:
        config_lines = config_file.read()
        for config_line in config_lines.split('\n'):
            config_fields = config_line.split(',')
            if len(config_fields) == 2 and config_fields[0] == 'do_save' and int(
                    config_fields[1]) != 0:
                args.do_save = True
            elif len(config_fields) == 2 and config_fields[0] == 'max_steps':
                args.max_steps = int(config_fields[1])
            elif len(config_fields) == 3 and config_fields[0] == 'lr_steps' and no_batch >= int(
                    config_fields[1]):
                args.lr = float(config_fields[2])
            elif len(config_fields) == 2 and config_fields[0] == 'verbose':
                args.verbose = int(config_fields[1])
            elif len(config_fields) == 2 and config_fields[0] == 'margin-a':
                args.margin_a = float(config_fields[1])
            elif len(config_fields) == 2 and config_fields[0] == 'margin-m':
                args.margin_m = float(config_fields[1])
            elif len(config_fields) == 2 and config_fields[0] == 'margin-b':
                args.margin_b = float(config_fields[1])
            elif len(config_fields) == 2 and config_fields[0] == 'do_save_threshold':
                args.do_save_threshold = float(config_fields[1])
    print(
        'batch: %d, save: %d, save_the: %.4f, max_steps: %d, lr: %.5f, verbose: %d, margin_a: %.2f, margin_m: %.2f, margin_b: %.2f' % \
        (no_batch, args.do_save, args.do_save_threshold, args.max_steps, args.lr, args.verbose,
         args.margin_a, args.margin_m, args.margin_b))



class AccMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(AccMetric, self).__init__(
            'acc', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []
        self.count = 0

    def update(self, labels, preds):
        self.count += 1
        preds = [preds[1]]  # use softmax output
        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.ndarray.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype('int32').flatten()
            label = label.asnumpy()
            if label.ndim == 2:
                label = label[:, 0]
            label = label.astype('int32').flatten()
            assert label.shape == pred_label.shape
            self.sum_metric += (pred_label.flat == label.flat).sum()
            self.num_inst += len(pred_label.flat)


class LossValueMetric(mx.metric.EvalMetric):
    def __init__(self):
        self.axis = 1
        super(LossValueMetric, self).__init__(
            'lossvalue', axis=self.axis,
            output_names=None, label_names=None)
        self.losses = []

    def update(self, labels, preds):
        loss = preds[-1].asnumpy()[0]
        self.sum_metric += loss
        self.num_inst += 1.0


def parse_args():
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--data-dir', default='', help='training set directory')
    parser.add_argument('--prefix', default='../models/model', help='directory to save model.')
    parser.add_argument('--pretrained', default='', help='pretrained model to load')
    parser.add_argument('--ckpt', type=int, default=1,
                        help='checkpoint saving option. 0: discard saving. 1: save when necessary. 2: always save')
    parser.add_argument('--loss-type', type=int, default=4, help='loss type')
    parser.add_argument('--verbose', type=int, default=2000,
                        help='do verification testing and model saving every verbose batches')
    parser.add_argument('--max-steps', type=int, default=0, help='max training batches')
    parser.add_argument('--end-epoch', type=int, default=100000, help='training epoch size.')
    parser.add_argument('--network', default='r50', help='specify network')
    parser.add_argument('--version-se', type=int, default=0, help='whether to use se in network')
    parser.add_argument('--version-input', type=int, default=1, help='network input config')
    parser.add_argument('--version-output', type=str, default='E',
                        help='network embedding output config')
    parser.add_argument('--version-unit', type=int, default=3, help='resnet unit config')
    parser.add_argument('--version-act', type=str, default='prelu',
                        help='network activation config')
    parser.add_argument('--use-deformable', type=int, default=0,
                        help='use deformable cnn in network')
    parser.add_argument('--lr', type=float, default=0.1, help='start learning rate')
    parser.add_argument('--lr-steps', type=str, default='', help='steps of lr changing')
    parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--fc7-wd-mult', type=float, default=1.0, help='weight decay mult for fc7')
    parser.add_argument('--bn-mom', type=float, default=0.9, help='bn mom')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--emb-size', type=int, default=512, help='embedding length')
    parser.add_argument('--per-batch-size', type=int, default=128,
                        help='batch size in each context')
    parser.add_argument('--margin-m', type=float, default=0.5, help='margin for loss')
    parser.add_argument('--margin-s', type=float, default=64.0, help='scale for feature')
    parser.add_argument('--margin-a', type=float, default=1.0, help='')
    parser.add_argument('--margin-b', type=float, default=0.0, help='')
    parser.add_argument('--easy-margin', type=int, default=0, help='')

    # sphereface
    parser.add_argument('--margin', type=int, default=4, help='margin for sphere')
    parser.add_argument('--beta', type=float, default=1000., help='param for sphere')
    parser.add_argument('--beta-min', type=float, default=5., help='param for sphere')
    parser.add_argument('--beta-freeze', type=int, default=0, help='param for sphere')
    parser.add_argument('--gamma', type=float, default=0.12, help='param for sphere')
    parser.add_argument('--power', type=float, default=1.0, help='param for sphere')
    parser.add_argument('--scale', type=float, default=0.9993, help='param for sphere')

    # triplet loss
    parser.add_argument('--center-alpha', type=float, default=0.5, help='')
    parser.add_argument('--center-scale', type=float, default=0.003, help='')
    parser.add_argument('--images-per-identity', type=int, default=0, help='')
    parser.add_argument('--triplet-bag-size', type=int, default=3600, help='')
    parser.add_argument('--triplet-alpha', type=float, default=0.3, help='')
    parser.add_argument('--triplet-max-ap', type=float, default=0.0, help='')

    parser.add_argument('--rand-mirror', type=int, default=1,
                        help='if do random mirror in training')
    parser.add_argument('--cutoff', type=int, default=0, help='cut off aug')
    parser.add_argument('--target', type=str, default='classify_megaface',
                        help='verification targets')

    parser.add_argument('--log_dir', type=str, default='/data/meiyi/insightface/output/log')
    parser.add_argument('--config_dir', type=str,
                        default='/home/meiyiyang/Face/insightface/output/config', help='')
    parser.add_argument('--result_dir', type=str,
                        default='/data/meiyi/insightface/output/result')
    parser.add_argument('--do_save', action='store_true',
                        help='true means save every model while training')
    parser.add_argument('--do_save_threshold', default=0.982, type=float)
    parser.add_argument('--tag', default='r50_v0', help='directory to save model.')
    parser.add_argument('--last_epoch', default=0, type=int, help='save model from last_epoch + 1')

    parser.add_argument('--fine_tune', type=int, default=0)
    args = parser.parse_args()
    return args


def get_symbol(args, arg_params, aux_params):
    # network
    if args.network[0] == 'd':
        embedding = fdensenet.get_symbol(args.emb_size, args.num_layers,
                                         version_se=args.version_se,
                                         version_input=args.version_input,
                                         version_output=args.version_output,
                                         version_unit=args.version_unit)
    elif args.network[0] == 'm':
        print('init mobilenet', args.num_layers)
        if args.num_layers == 1:
            embedding = fmobilenet.get_symbol(args.emb_size,
                                              version_se=args.version_se,
                                              version_input=args.version_input,
                                              version_output=args.version_output,
                                              version_unit=args.version_unit)
        elif args.num_layers == 2:
            embedding = fmobilenetv2.get_symbol(args.emb_size)
        else:
            embedding = fmnasnet.get_symbol(args.emb_size)

    elif args.network[0] == 'i':
        print('init inception-resnet-v2', args.num_layers)
        embedding = finception_resnet_v2.get_symbol(args.emb_size,
                                                    version_se=args.version_se,
                                                    version_input=args.version_input,
                                                    version_output=args.version_output,
                                                    version_unit=args.version_unit)
    elif args.network[0] == 'x':
        print('init xception', args.num_layers)
        embedding = fxception.get_symbol(args.emb_size,
                                         version_se=args.version_se,
                                         version_input=args.version_input,
                                         version_output=args.version_output,
                                         version_unit=args.version_unit)
    elif args.network[0] == 'p':
        print('init dpn', args.num_layers)
        embedding = fdpn.get_symbol(args.emb_size, args.num_layers,
                                    version_se=args.version_se, version_input=args.version_input,
                                    version_output=args.version_output,
                                    version_unit=args.version_unit)
    elif args.network[0] == 'n':
        print('init nasnet', args.num_layers)
        if args.num_layers == 1:
            embedding = fnasnet.get_symbol(args.emb_size)
        else:
            embedding = fmnasnet.get_symbol(args.emb_size)
    elif args.network[0] == 's':
        print('init spherenet', args.num_layers)
        embedding = spherenet.get_symbol(args.emb_size, args.num_layers)
    elif args.network[0] == 'y':
        print('init mobilefacenet', args.num_layers)
        embedding = fmobilefacenet.get_symbol(args.emb_size, bn_mom=args.bn_mom,
                                              wd_mult=args.fc7_wd_mult)
    else:
        print('init resnet', args.num_layers)
        embedding = fresnet.get_symbol(args.emb_size, args.num_layers,
                                       version_se=args.version_se, version_input=args.version_input,
                                       version_output=args.version_output,
                                       version_unit=args.version_unit,
                                       version_act=args.version_act)

    all_label = mx.symbol.Variable('softmax_label')
    gt_label = all_label
    _weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0,
                                 wd_mult=args.fc7_wd_mult)

    # loss
    extra_loss = None
    if args.loss_type == 0:  # softmax
        _bias = mx.symbol.Variable('fc7_bias', lr_mult=2.0, wd_mult=0.0)
        fc7 = mx.sym.FullyConnected(data=embedding, weight=_weight, bias=_bias,
                                    num_hidden=args.num_classes, name='fc7')
    elif args.loss_type == 1:  # sphere
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        fc7 = mx.sym.LSoftmax(data=embedding, label=gt_label, num_hidden=args.num_classes,
                              weight=_weight,
                              beta=args.beta, margin=args.margin, scale=args.scale,
                              beta_min=args.beta_min, verbose=1000, name='fc7')
    elif args.loss_type == 2:
        s = args.margin_s
        m = args.margin_m
        assert (s > 0.0)
        assert (m > 0.0)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True,
                                    num_hidden=args.num_classes, name='fc7')
        s_m = s * m
        gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=s_m, off_value=0.0)
        fc7 = fc7 - gt_one_hot
    elif args.loss_type == 4:
        s = args.margin_s
        m = args.margin_m
        assert s > 0.0
        assert m >= 0.0
        assert m < (math.pi / 2)
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True,
                                    num_hidden=args.num_classes, name='fc7')
        zy = mx.sym.pick(fc7, gt_label, axis=1)
        cos_t = zy / s
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = math.sin(math.pi - m) * m
        # threshold = 0.0
        threshold = math.cos(math.pi - m)
        if args.easy_margin:
            cond = mx.symbol.Activation(data=cos_t, act_type='relu')
        else:
            cond_v = cos_t - threshold
            cond = mx.symbol.Activation(data=cond_v, act_type='relu')
        body = cos_t * cos_t
        body = 1.0 - body
        sin_t = mx.sym.sqrt(body)
        new_zy = cos_t * cos_m
        b = sin_t * sin_m
        new_zy = new_zy - b
        new_zy = new_zy * s
        if args.easy_margin:
            zy_keep = zy
        else:
            zy_keep = zy - s * mm
        new_zy = mx.sym.where(cond, new_zy, zy_keep)

        diff = new_zy - zy
        diff = mx.sym.expand_dims(diff, 1)
        gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=1.0, off_value=0.0)
        body = mx.sym.broadcast_mul(gt_one_hot, diff)
        fc7 = fc7 + body
    elif args.loss_type == 5:
        s = args.margin_s
        m = args.margin_m
        assert s > 0.0
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n') * s
        fc7 = mx.sym.FullyConnected(data=nembedding, weight=_weight, no_bias=True,
                                    num_hidden=args.num_classes, name='fc7')
        if args.margin_a != 1.0 or args.margin_m != 0.0 or args.margin_b != 0.0:
            if args.margin_a == 1.0 and args.margin_m == 0.0:
                s_m = s * args.margin_b
                gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=s_m,
                                            off_value=0.0)
                fc7 = fc7 - gt_one_hot
            else:
                zy = mx.sym.pick(fc7, gt_label, axis=1)
                cos_t = zy / s
                t = mx.sym.arccos(cos_t)
                if args.margin_a != 1.0:
                    t = t * args.margin_a
                if args.margin_m > 0.0:
                    t = t + args.margin_m
                body = mx.sym.cos(t)
                if args.margin_b > 0.0:
                    body = body - args.margin_b
                new_zy = body * s
                diff = new_zy - zy
                diff = mx.sym.expand_dims(diff, 1)
                gt_one_hot = mx.sym.one_hot(gt_label, depth=args.num_classes, on_value=1.0,
                                            off_value=0.0)
                body = mx.sym.broadcast_mul(gt_one_hot, diff)
                fc7 = fc7 + body
    elif args.loss_type == 12:  # triplet loss
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n')
        anchor = mx.symbol.slice_axis(nembedding, axis=0, begin=0, end=args.per_batch_size // 3)
        positive = mx.symbol.slice_axis(nembedding, axis=0, begin=args.per_batch_size // 3,
                                        end=2 * args.per_batch_size // 3)
        negative = mx.symbol.slice_axis(nembedding, axis=0, begin=2 * args.per_batch_size // 3,
                                        end=args.per_batch_size)
        ap = anchor - positive
        an = anchor - negative
        ap = ap * ap
        an = an * an
        ap = mx.symbol.sum(ap, axis=1, keepdims=1)  # (T,1)
        an = mx.symbol.sum(an, axis=1, keepdims=1)  # (T,1)
        triplet_loss = mx.symbol.Activation(data=(ap - an + args.triplet_alpha), act_type='relu')
        triplet_loss = mx.symbol.mean(triplet_loss)
        # triplet_loss = mx.symbol.sum(triplet_loss)/(args.per_batch_size//3)
        extra_loss = mx.symbol.MakeLoss(triplet_loss)

    out_list = [mx.symbol.BlockGrad(embedding)]
    softmax = None
    if args.loss_type < 10:
        softmax = mx.symbol.SoftmaxOutput(data=fc7, label=gt_label, name='softmax',
                                          normalization='valid')
        out_list.append(softmax)
    if softmax is None:
        out_list.append(mx.sym.BlockGrad(gt_label))
    if extra_loss is not None:
        out_list.append(extra_loss)

    out = mx.symbol.Group(out_list)

    # fine-tune
    if args.fine_tune:
        arg_params = dict({k: arg_params[k] for k in arg_params if 'fc7' not in k})

    return (out, arg_params, aux_params)


def train_net(args):
    # Set training parameters (do_save, max_step, lr)
    config_path = os.path.join(args.config_dir, 'config_' + args.tag + '.txt')
    if os.path.exists(config_path):
        get_config(config_path, int(args.last_epoch) * args.verbose, args)
        
    ctx = []
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()
    else:
        cvd = []
    if len(cvd) > 0:
        for i in xrange(len(cvd.split(','))):
            ctx.append(mx.gpu(i))
    if len(ctx) == 0:
        ctx = [mx.cpu()]
        print('use cpu')
    else:
        print('gpu num:', len(ctx))
    prefix = args.prefix
    prefix_dir = os.path.dirname(prefix)
    if not os.path.exists(prefix_dir):
        os.makedirs(prefix_dir)
    end_epoch = args.end_epoch
    args.ctx_num = len(ctx)
    args.num_layers = int(args.network[1:])
    print('num_layers', args.num_layers)
    if args.per_batch_size == 0:
        args.per_batch_size = 128
    args.batch_size = args.per_batch_size * args.ctx_num
    args.rescale_threshold = 0
    args.image_channel = 3

    os.environ['BETA'] = str(args.beta)
    data_dir_list = args.data_dir.split(',')
    assert len(data_dir_list) == 1
    data_dir = data_dir_list[0]

    prop = face_image.load_property(data_dir)
    args.num_classes = prop.num_classes
    image_size = prop.image_size
    args.image_h = image_size[0]
    args.image_w = image_size[1]
    print('image_size', image_size)
    assert (args.num_classes > 0)
    print('num_classes', args.num_classes)
    path_imgrec = os.path.join(data_dir, "train.rec")

    if args.loss_type == 1 and args.num_classes > 20000:
        args.beta_freeze = 5000
        args.gamma = 0.06

    if args.loss_type < 9:
        assert args.images_per_identity == 0
    else:
        if args.images_per_identity == 0:
            if args.loss_type == 11:
                args.images_per_identity = 2
            elif args.loss_type == 10 or args.loss_type == 9:
                args.images_per_identity = 16
            elif args.loss_type == 12 or args.loss_type == 13:
                args.images_per_identity = 5
                assert args.per_batch_size % 3 == 0
        assert args.images_per_identity >= 2
        args.per_identities = int(args.per_batch_size / args.images_per_identity)

    print('Called with argument:', args)

    data_shape = (args.image_channel, image_size[0], image_size[1])
    mean = None

    begin_epoch = 0
    base_lr = args.lr
    base_wd = args.wd
    base_mom = args.mom
    if len(args.pretrained) == 0:
        arg_params = None
        aux_params = None
        sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
    else:
        vec = args.pretrained.split(',')
        print('loading', vec)
        _, arg_params, aux_params = mx.model.load_checkpoint(vec[0], int(vec[1]))
        sym, arg_params, aux_params = get_symbol(args, arg_params, aux_params)
    if args.network[0] == 's':
        data_shape_dict = {'data': (args.per_batch_size,) + data_shape}
        spherenet.init_weights(sym, data_shape_dict, args.num_layers)

    triplet_params = None
    if args.loss_type == 12 or args.loss_type == 13:
        triplet_params = [args.triplet_bag_size, args.triplet_alpha, args.triplet_max_ap]

    if args.fine_tune:
        # fixed_params = sym.list_arguments()[:239]
        # fixed_params = sym.list_arguments()[:239]
        fixed_params = dict({k: arg_params[k] for k in arg_params if 'fc7' not in k})
        model = mx.mod.Module(context=ctx, symbol=sym, fixed_param_names=fixed_params)
    else:
        model = mx.mod.Module(
            context=ctx,
            symbol=sym,
        )
        
    val_dataiter = None

    if len(data_dir_list) == 1 and args.loss_type != 12 and args.loss_type != 13:
        train_dataiter = FaceImageIter(
            batch_size=args.batch_size,
            data_shape=data_shape,
            path_imgrec=path_imgrec,
            shuffle=True,
            rand_mirror=args.rand_mirror,
            mean=mean,
            cutoff=args.cutoff,
        )
    else:
        iter_list = []
        for _data_dir in data_dir_list:
            _path_imgrec = os.path.join(_data_dir, "train.rec")
            _dataiter = FaceImageIter(
                batch_size=args.batch_size,
                data_shape=data_shape,
                path_imgrec=_path_imgrec,
                shuffle=True,
                rand_mirror=args.rand_mirror,
                mean=mean,
                cutoff=args.cutoff,
                images_per_identity=args.images_per_identity,
                triplet_params=triplet_params,
            )
            iter_list.append(_dataiter)
        iter_list.append(_dataiter)
        train_dataiter = FaceImageIterList(iter_list)

    if args.loss_type < 10:
        _metric = AccMetric()
    else:
        _metric = LossValueMetric()
    eval_metrics = [mx.metric.create(_metric)]

    if args.network[0] == 'r' or args.network[0] == 'y':
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="out",
                                     magnitude=2)  # resnet style
    elif args.network[0] == 'i' or args.network[0] == 'x':
        initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                                     magnitude=2)  # inception
    else:
        initializer = mx.init.Xavier(rnd_type='uniform', factor_type="in", magnitude=2)
    _rescale = 1.0 / args.ctx_num
    opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    som = 20
    if args.loss_type == 12 or args.loss_type == 13:
        som = 2
    _cb = mx.callback.Speedometer(args.batch_size, som)

    # meiyi change
    result_path = os.path.join(args.result_dir, "result_" + args.tag + ".txt")
    with open(result_path, 'a+') as result_file:
        result_file.write('\n')
        result_file.write('date_dir: %s\n' % args.data_dir)
        result_file.write('prefix: %s\n' % args.prefix)
        result_file.write('tag: %s\n' % args.tag)
        result_file.write('pretrain: %s\n' % args.pretrained)
        result_file.write('per_batch_size: %d\n' % args.per_batch_size)
        result_file.write(
            'save: %d, max_steps: %d, lr: %.5f, verbose: %d, margin_a: %.2f, margin_m: %.2f, margin_b: %.2f\n'
            % (args.do_save, args.max_steps, args.lr, args.verbose, args.margin_a, args.margin_m,
               args.margin_b))

    def ver_test(nbatch, msave):
        results = []
        result_path = os.path.join(args.result_dir, "result_" + args.tag + ".txt")
        with open(result_path, 'a+') as result_file:
            current_time = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
            result_file.write('%s\t%d\t%d\t%1.4f\t' % (current_time, nbatch, msave, opt.lr))

            for name in args.target.split(','):
                path = os.path.join(data_dir, name + ".bin")
                if os.path.exists(path):
                    mode_classify = 0  # 0 normal(lfw), 1 classify, 2 classify megaface
                    if "megaface" in name:
                        mode_classify = 2
                    elif "classify_suning_test" in name:
                        mode_classify = 3
                    elif "classify" in name:
                        mode_classify = 1

                    if mode_classify != 2:
                        dataset = verification_mxnet_meiyi.load_bin(path, image_size)
                    else:
                        dataset = verification_mxnet_meiyi.load_megaface_bin(path, image_size)

                    val1, std1, acc2, std2, xnorm, embeddings_list = verification_mxnet_meiyi.test(
                        dataset, model, args.batch_size, 10, None, None, mode_classify)
                    print('batch: %d\t%s\tacc: %1.5f+-%1.5f\tval: %1.5f+-%1.5f' % (
                    nbatch, name, acc2, std2, val1, std1))
                    result_file.write(
                        '%s\t%1.5f\t%1.5f\t%1.5f\t%1.5f\t' % (name, acc2, std2, val1, std1))
                    results.append(acc2)

                    # add summary
                    summary_dir = os.path.join(args.result_dir, args.tag)
                    if not os.path.exists(summary_dir):
                        os.makedirs(summary_dir)

                    with SummaryWriter(summary_dir) as sw:
                        sw.add_scalar(tag='eval/acc_' + name, value=acc2, global_step=nbatch)
                        sw.add_scalar(tag='eval/val_' + name, value=val1, global_step=nbatch)
                        sw.add_scalar(tag='learning_rate', value=args.lr, global_step=nbatch)
                        print("write summary to ", args.result_dir)

            result_file.write('\n')
        return results
    # meiyi change END

    highest_acc = [0.0, 0.0]  # lfw and target
    # for i in xrange(len(ver_list)):
    #  highest_acc.append(0.0)
    global_step = [int(args.last_epoch) * args.verbose]
    save_step = [int(args.last_epoch)]
    if len(args.lr_steps) == 0:
        lr_steps = [40000, 60000, 80000]
        if args.loss_type >= 1 and args.loss_type <= 7:
            lr_steps = [100000, 140000, 160000]
        p = 512.0 / args.batch_size
        for l in xrange(len(lr_steps)):
            lr_steps[l] = int(lr_steps[l] * p)
    else:
        lr_steps = [int(x) for x in args.lr_steps.split(',')]
    print('lr_steps', lr_steps)

    def _batch_callback(param):
        # global global_step
        global_step[0] += 1
        mbatch = global_step[0]

        _cb(param)

        if mbatch >= 0 and mbatch % args.verbose == 0:
            # update args
            if mbatch == args.verbose:
                result_path = os.path.join(args.result_dir, "result_" + args.tag + ".txt")
                with open(result_path, 'a+') as result_file:
                    result_file.write('\n')
                    args_dict = args.__dict__
                    for key in args_dict:
                        result_file.write("%s: %s\n" % (key, str(args_dict[key])))
                    result_file.write('\n')


            # update config
            config_path = os.path.join(args.config_dir, 'config_' + args.tag + '.txt')
            if os.path.exists(config_path):
                get_config(config_path, mbatch, args)
                opt.lr = args.lr

            save_step[0] += 1
            msave = save_step[0]
            acc_list = ver_test(mbatch, msave)

            if args.do_save:
                arg, aux = model.get_params()
                mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
            elif acc_list[0] >= args.do_save_threshold:
                arg, aux = model.get_params()
                mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)
            elif len(acc_list) > 0 and acc_list[0] >= highest_acc[0]:
                arg, aux = model.get_params()
                mx.model.save_checkpoint(prefix, msave, model.symbol, arg, aux)

            # update highest
            if len(acc_list) > 0:
                highest_acc[0] = max(highest_acc[0], acc_list[0])
            print('[%d]Accuracy-Highest: %1.5f' % (mbatch, highest_acc[0]))

        if mbatch <= args.beta_freeze:
            _beta = args.beta
        else:
            move = max(0, mbatch - args.beta_freeze)
            _beta = max(args.beta_min,
                        args.beta * math.pow(1 + args.gamma * move, -1.0 * args.power))
        # print('beta', _beta)
        os.environ['BETA'] = str(_beta)
        if args.max_steps > 0 and mbatch > args.max_steps:
            sys.exit(0)

    epoch_cb = None
    train_dataiter = mx.io.PrefetchingIter(train_dataiter)

    model.fit(train_dataiter,
              begin_epoch=begin_epoch,
              num_epoch=end_epoch,
              eval_data=val_dataiter,
              eval_metric=eval_metrics,
              kvstore='local',
              optimizer=opt,
              # optimizer_params   = optimizer_params,
              initializer=initializer,
              arg_params=arg_params,
              aux_params=aux_params,
              allow_missing=True,
              batch_end_callback=_batch_callback,
              epoch_end_callback=epoch_cb)


def main():
    # time.sleep(3600*6.5)
    global args
    args = parse_args()
    set_log_config(os.path.join(args.log_dir, 'log_' + args.tag + '.txt'))

    train_net(args)


if __name__ == '__main__':
    main()

