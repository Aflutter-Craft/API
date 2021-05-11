import os

from net import *


# set up device
DEVICE = torch.device("cuda" if torch.cuda.is_available()  # type: ignore
                      else "cpu")

# on floydhub the models are mounted as a dataset to /models
MODELS_PATH = '/models' if os.path.exists("/models") else 'models'


# load pretrained models
def load_models():
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

    return vgg, samodule, decoder


# perform style transfer
def style_transfer(vgg, decoder, sa_module, content, style, alpha=1):
    assert (0.0 <= alpha <= 1.0)  # make sure alpha value is valid

    # move samodule and decoder to gpu (no need to move vgg since we dont use it all)
    sa_module.to(DEVICE)
    decoder.to(DEVICE)

    # move content and style to device as well
    content = content.to(DEVICE)
    style = style.to(DEVICE)

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
