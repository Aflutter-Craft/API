An API for style transfer build using Flask + Swagger.

## Features

- accepts style image as URL to one of the style images in S3
- accepts style image as a base64 encoded image
- accepts style content trade-off value as alpha

## Running locally

- install requirements with `pip install -r requirements.txt`
- for auto reload set these variables in your shell `export FLASK_APP=app.py && export FLASK_ENV=development`
- start API server with `flask run`
- navigate to `http://127.0.0.1:5000/docs/` to test from web.

## API docs

![docs](screens/docs.png)

## General Workflow

![workflow](screens/workflow.png)
