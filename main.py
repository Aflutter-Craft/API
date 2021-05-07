import io
import os
import time
import urllib.request as request

from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.datastructures import UploadFile
from fastapi.params import File
from starlette.responses import FileResponse, HTMLResponse
import torch
from torchvision.utils import save_image

from net import *

# Globals
app = FastAPI()
DEVICE = torch.device("cuda" if torch.cuda.is_available()  # type: ignore
                      else "cpu")
MODELS_PATH = 'models'
OUTPUT_FOLDER = 'results'
if not os.path.exists(OUTPUT_FOLDER):
    os.mkdir(OUTPUT_FOLDER)



# perform style transfer
def style_transfer(vgg, decoder, sa_module, content, style, alpha=1):
    assert (0.0 <= alpha <= 1.0)  # make sure alpha value is valid

    # move samodule and decoder to gpu (no need to move vgg since we dont use it all)
    sa_module.to(DEVICE)
    decoder.to(DEVICE)

    # get features for both style and content
    Content4_1, Content5_1, Style4_1, Style5_1 = feat_extractor(
        vgg, content, style, DEVICE)

    # get content features importance (used to select how much of content to keep according to alpha)
    Fccc = sa_module(Content4_1, Content4_1, Content5_1, Content5_1)

    # get final image features
    feat = sa_module(Content4_1, Style4_1, Content5_1, Style5_1)
    # change final image according to alpha value
    feat = feat * alpha + Fccc * (1 - alpha)
    # return decoded final image
    return decoder(feat)


@app.get('/')
def home() -> HTMLResponse:
    return HTMLResponse("use /style API route for styling images")


@app.post('/style')
def style_image(content_img: UploadFile = File(...), style_img: UploadFile = File(b""),  # type: ignore
                style_path: str = '', alpha: float = 0.8) -> FileResponse:

    # check if both style_img and style_path are empty
    if style_path == '' and style_img == b"":
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

    # prepare models
    decoder = decoder_arch
    samodule = SAModule(in_dim=512)
    vgg = vgg_arch

    # set to evalaution mode (faster)
    decoder.eval()
    samodule.eval()
    vgg.eval()

    # load saved model states
    decoder.load_state_dict(torch.load(f'{MODELS_PATH}/decoder.pth'))
    samodule.load_state_dict(torch.load(f'{MODELS_PATH}/transformer.pth'))
    vgg.load_state_dict(torch.load(f'{MODELS_PATH}/vgg.pth'))

    # finally perform style transfer without tracking gradients
    with torch.no_grad():
        output = style_transfer(vgg, decoder, samodule,
                                content, style, alpha)

    # save output image
    out_name = f'{OUTPUT_FOLDER}/result_{time.time()}_{alpha}.jpg'
    save_image(output, out_name)

    return FileResponse(f'{OUTPUT_FOLDER}/{out_name}')
