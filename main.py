import io
import logging
import os

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from flask_httpauth import HTTPBasicAuth
from flask import Flask, make_response, request, jsonify
from werkzeug.utils import secure_filename
from prometheus_flask_exporter import PrometheusMetrics
from functools import wraps


# UPLOAD_FOLDER = '/workspace/reference_voices'
UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = {'wav', 'txt'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config.from_prefixed_env('CHATTERBOX_API')

auth = HTTPBasicAuth()
metrics = PrometheusMetrics(app, metrics_decorator=auth.login_required)

# if "FORCE_CUDA" in app.config and app.config["FORCE_CUDA"]:
#     logging.info('Forcing GPU')
#     device = 'cuda'
# elif torch.cuda.is_available():
#     logging.info('Using GPU')
#     device = 'cuda'
# elif torch.backends.mps.is_available():
#     #Mac with M1/M2/M3/M4
#     logging.info('Using CPU (Apple Silicon)')
#     device = 'mps'
# else:
#     logging.info('Using CPU')
#     device = 'cpu'

model: "ChatterboxTTS" = ChatterboxTTS.from_pretrained(device="cuda")
model.t3._step_compilation_target = torch.compile(
    model.t3._step_compilation_target, fullgraph=True, backend="cudagraphs"
)

# text = "Today is the day. I want to move like a titan at dawn, sweat like a god forging lightning. No more excuses. From now on, my mornings will be temples of discipline. I am going to work out like the gods… every damn day."
#
# # If you want to synthesize with a different voice, specify the audio prompt
# # AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
# print("Generating!")


@auth.verify_password
def flask_verify_pw(username, password):
    return check_credentials(username, password)

def check_credentials(username, password):
    username_check = not app.config['BASIC_AUTH_USERNAME'] or username == app.config['BASIC_AUTH_USERNAME']
    password_check = not app.config['BASIC_AUTH_PASSWORD'] or password == app.config['BASIC_AUTH_PASSWORD']
    return username_check and password_check

def basic_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_credentials(auth.username, auth.password):
            return jsonify({'message': 'Unauthorized'}), 401
        return f(*args, **kwargs)

    return decorated_function


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_voice', methods=['POST'])
@basic_auth
def upload_voice():
    if 'file' not in request.files:
        return jsonify({'message': 'Bad Request; file not present'}), 400
    voice_file = request.files['file']
    if voice_file.filename == '':
        return jsonify({'message': 'Bad Request; file name not present'}), 400
    if voice_file and allowed_file(voice_file.filename):
        filename = secure_filename(voice_file.filename)
        voice_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        if 'reference_text' in request.files:
            reference_text_file = request.files['reference_text']
            reference_text_filename = voice_file.filename.replace(".wav", ".txt")
            reference_text_file.save(os.path.join(app.config['UPLOAD_FOLDER'], reference_text_filename))
    return jsonify({'message': 'OK'}), 200


@app.route('/generate', methods=['POST'])
@basic_auth
def generate():
    text = request.json['text']
    voice_name = str(request.json['voice_name'])
    exaggeration = str(request.json['exaggeration'])
    if not exaggeration:
        exaggeration = 0.5
    cfg_weight = str(request.json['cfg_weight'])
    if not cfg_weight:
        cfg_weight = 0.5

    logging.debug("text: {}, voice_name: {}, exaggeration: {}, cfg_weight: {}".format(text, voice_name, exaggeration, cfg_weight))

    # reference_text_path = os.path.join(app.config['UPLOAD_FOLDER'], voice_name.replace(".wav", ".txt"))
    # reference_text = ""
    # if os.path.isfile(reference_text_path):
    #     f = open(reference_text_path, "r")
    #     reference_text = f.read()
    #     f.close()

    voice_path = app.config['UPLOAD_FOLDER'] + "/" + voice_name
    try:
        wav = model.generate(
            text,
            audio_prompt_path=voice_path,
            exaggeration=0.5,
            cfg_weight=0.5
        )

        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)


        response = make_response(buffer.read())
        response.headers['Content-Type'] = 'audio/wav'
        response.headers['Content-Disposition'] = 'attachment; filename=sound.wav'
        return response
    except Exception as e:
        return jsonify({"message": f"tts failed", "Exception": str(e)}), 400


if __name__ == '__main__':

    app.run(host='0.0.0.0',port=5555,  debug=True)