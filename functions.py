import os

import torch
import torch.nn as nn
from torchvision import transforms

# vars
# allowed files extentions
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])

# create output dir if it doesnt exist
OUTPUT_FOLDER = 'results'
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)

# make sure styles folder exists
STYLES_DIR = 'styles'
if not os.path.exists(STYLES_DIR):
    os.mkdir(STYLES_DIR)


# check if a file meets allowed extentions
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# loss functions
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


# transforms to perfom on image before using
def test_transform():
    transform_list = []
    transform_list.append(transforms.Resize(512))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


# extract style and content features
def feat_extractor(vgg, content, style, DEVICE):
    # extract used layers from vgg network
    enc_1 = nn.Sequential(*list(vgg.children())[:4])  # input -> relu1_1
    enc_2 = nn.Sequential(*list(vgg.children())[4:11])  # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*list(vgg.children())[11:18])  # relu2_1 -> relu3_1
    enc_4 = nn.Sequential(*list(vgg.children())[18:31])  # relu3_1 -> relu4_1
    enc_5 = nn.Sequential(*list(vgg.children())[31:44])  # relu4_1 -> relu5_1

    # move everything to GPU
    enc_1.to(DEVICE)
    enc_2.to(DEVICE)
    enc_3.to(DEVICE)
    enc_4.to(DEVICE)
    enc_5.to(DEVICE)

    # extract content features
    Content4_1 = enc_4(enc_3(enc_2(enc_1(content))))
    Content5_1 = enc_5(Content4_1)
    # extract style features
    Style4_1 = enc_4(enc_3(enc_2(enc_1(style))))
    Style5_1 = enc_5(Style4_1)

    return Content4_1, Content5_1, Style4_1, Style5_1
