import io
import os
import time
import urllib.request as request

from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.datastructures import UploadFile
from fastapi.params import File
from starlette.responses import FileResponse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image


# Globals
app = FastAPI()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore
MODELS_PATH = 'models'
OUTPUT_FOLDER = 'results'
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)


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


# Networks
decoder_arch = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)


vgg_arch = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


# Self attention network
class SANet(nn.Module):
    def __init__(self, in_dim):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.g = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.h = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        O = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content.size()
        O = O.view(b, c, h, w)
        O = self.out_conv(O)
        O += content
        return O


# self attention module with attention networks for relu4_1, relu5_1
class SAModule(nn.Module):
    def __init__(self, in_dim):
        super(SAModule, self).__init__()
        self.sanet4_1 = SANet(in_dim=in_dim)
        self.sanet5_1 = SANet(in_dim=in_dim)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_dim, in_dim, (3, 3))

    def forward(self, content4_1, style4_1, content5_1, style5_1):
        return self.merge_conv(self.merge_conv_pad(self.sanet4_1(content4_1, style4_1) + self.upsample5_1(self.sanet5_1(content5_1, style5_1))))


# transforms to perfom on image before using
def test_transform():
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Resize(512))
    transform = transforms.Compose(transform_list)
    return transform


# extract style and content features
def feat_extractor(vgg, content, style):
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


# perform style transfer
def style_transfer(vgg, decoder, sa_module, content, style, alpha=1):
    assert (0.0 <= alpha <= 1.0)  # make sure alpha value is valid

    # move samodule and decoder to gpu (no need to move vgg since we dont use it all)
    sa_module.to(DEVICE)
    decoder.to(DEVICE)

    # get features for both style and content
    Content4_1, Content5_1, Style4_1, Style5_1 = feat_extractor(
        vgg, content, style)

    # get content features importance (used to select how much of content to keep according to alpha)
    Fccc = sa_module(Content4_1, Content4_1, Content5_1, Content5_1)

    # get final image features
    feat = sa_module(Content4_1, Style4_1, Content5_1, Style5_1)
    # change final image according to alpha value
    feat = feat * alpha + Fccc * (1 - alpha)
    # return decoded final image
    return decoder(feat)


@app.post('/style')
def style_image(content_img: UploadFile = File(...), style_img: UploadFile = File(b""),  # type: ignore
                style_path: str = '', alpha: float = 0.8) -> FileResponse:

    # check if both style_img and style_path are empty
    if style_path == '' or style_img == b"":
        raise HTTPException(
            status_code=400, detail="Either Style image or Style path must be provided!")

    # get style image if path was passed otherwise read sent image bytes
    if style_path != '':
        # style path format: style_type/image_name.png
        # prefix url
        url = "https://aflutter-craft.s3.amazonaws.com/styles/"
        file_name = style_path.split('/')[1]  # get image name

        # check if file exists first before downloading it again
        if os.path.isfile(f'styles/{file_name}'):
            pass
        else:
            request.urlretrieve(f'{url}{style_path}', f'styles/{file_name}')
        style = Image.open(f'styles/{file_name}').convert('RGB')
    else:
        style = Image.open(io.BytesIO(
            style_img.file.read())).convert('RGB')

    # apply transformations
    trans = test_transform()
    content = io.BytesIO(content_img.file.read())  # read image as bytes
    content = trans(Image.open(content).convert('RGB')).unsqueeze(0)
    style = trans(style).unsqueeze(0)

    # finally perform style transfer without tracking gradients
    with torch.no_grad():
        output = style_transfer(vgg, decoder, samodule,
                                content, style, alpha)

    out_name = f'{OUTPUT_FOLDER}/result_{time.time()}_{alpha}.jpg'
    save_image(output, out_name)

    return FileResponse(out_name)
