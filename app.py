import os
import time
import urllib.request as urlrequest

from PIL import Image
from flasgger import Swagger, swag_from
from flask import Flask, request, send_file
import torch
from torchvision.utils import save_image
from werkzeug.exceptions import BadRequest

from net import *
from style import *


# flask application
app = Flask(__name__)

# docs builder config
swagger_config = {
    "headers": [
    ],
    "specs": [
        {
            "endpoint": 'apispec_1',
            "route": '/apispec_1.json',
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}
swagger_template = {
    "info": {
        "title": "Aflutter Craft",
        "description": "API for neural style transfer using style attentional networks",
        "contact": {
            "email": "abubakaryagob@gmail.com",
            "url": "https://blacksuan19.tk",
        },
        "version": "0.0.5"
    },
}

# set page title
app.config['SWAGGER'] = {
    'title': 'Aflutter Craft',
}
swagger = Swagger(app, config=swagger_config, template=swagger_template)


@app.route('/style', methods=["POST"])
@swag_from("api_docs.yml")  # docs
def style_image():
    content_img = request.files.get('content_img')
    style_img = request.files.get('style_img')
    style_path = request.form.get('style_path')
    alpha = request.form.get('alpha')
    alpha = float(alpha)  # type: ignore

    # alpha value validation
    if alpha > 1.0 or alpha < 0:
        return BadRequest("Alpha value should be 0 < alpha < 1")

    # make sure content is present and valid
    if not content_img:
        return BadRequest("content image is required!")
    if content_img.filename == '':
        return BadRequest("File name is not present in request")
    if not allowed_file(content_img.filename):
        return BadRequest("Invalid content image file type")

    # style validation
    if not style_img and not style_path:
        return BadRequest("either custom style image or style image path are required!")
    if style_img and style_path:
        return BadRequest("only one of style path or style images should be provided!")

    # get style image if path was passed otherwise read sent image bytes
    if style_path:
        # style path format: style_type/image_name.png
        # prefix url
        url = "https://aflutter-craft.s3.amazonaws.com/styles/"
        file_name = style_path.split('/')[1]  # get image name

        # check if file exists first before downloading it again
        if os.path.isfile(f'styles/{file_name}'):
            pass
        else:
            urlrequest.urlretrieve(f'{url}{style_path}', f'styles/{file_name}')
        style = Image.open(f'styles/{file_name}').convert('RGB')
    else:
        if not allowed_file(style_img.filename):
            return BadRequest("Invalid style image file type")
        style = Image.open(style_img).convert('RGB')

    # apply transformations
    trans = test_transform()
    content = trans(Image.open(content_img).convert('RGB')).unsqueeze(0)
    style = trans(style).unsqueeze(0)

    # finally perform style transfer without tracking gradients
    vgg, samodule, decoder = load_models()
    with torch.no_grad():
        output = style_transfer(vgg, decoder, samodule,
                                content, style, alpha)

    # save output image
    out_name = f'{OUTPUT_FOLDER}/result_{time.time()}_{alpha}.jpg'
    save_image(output, out_name)

    return send_file(out_name, mimetype='image/jpg')


# only used for local developmenet and testing
if __name__ == "__main__":
    print("* Starting web server... please wait until server has fully started")
    app.run(host='0.0.0.0', threaded=False)
