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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logger.info("Initializing Chatterbox TTS")
tts_model: "ChatterboxTTS" = ChatterboxTTS.from_pretrained(device="cuda")


dtype=torch.bfloat16

tts_model.t3.to(dtype=dtype)
tts_model.conds.t3.to(dtype=dtype)
torch.cuda.empty_cache()
logger.info("Compilation...")
tts_model.t3._step_compilation_target = torch.compile(
    tts_model.t3._step_compilation_target, fullgraph=True, backend="cudagraphs"
)
logger.info("Compilation done")

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
    temperature = str(request.json['temperature'])
    if not temperature:
        temperature = 0.8
    tokens_per_slice = str(request.json['tokens_per_slice'])
    if not tokens_per_slice:
        tokens_per_slice = None
    remove_milliseconds = str(request.json['remove_milliseconds'])
    if not remove_milliseconds:
        remove_milliseconds = None
    remove_milliseconds_start = str(request.json['remove_milliseconds_start'])
    if not remove_milliseconds_start:
        remove_milliseconds_start = None
    max_new_tokens = str(request.json['max_new_tokens'])
    if not max_new_tokens:
        max_new_tokens = None
    max_cache_len = str(request.json['max_cache_len'])
    if not max_cache_len:
        max_cache_len = 1500
    repetition_penalty = str(request.json['repetition_penalty'])
    if not repetition_penalty:
        repetition_penalty = 1.2
    min_p = str(request.json['min_p'])
    if not min_p:
        min_p = 0.05
    top_p = str(request.json['top_p'])
    if not top_p:
        top_p = 1.0

    logger.debug("text: {}, voice_name: {}, exaggeration: {}, cfg_weight: {}".format(text, voice_name, exaggeration, cfg_weight))

    voice_path = app.config['UPLOAD_FOLDER'] + "/" + voice_name
    try:
        tts_model.prepare_conditionals(voice_path)
        tts_model.conds.t3.to(dtype=dtype)
        wav = tts_model.generate(
            text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            tokens_per_slice=tokens_per_slice,
            remove_milliseconds=remove_milliseconds,
            remove_milliseconds_start=remove_milliseconds_start,
            max_new_tokens=max_new_tokens,
            max_cache_len=max_cache_len,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )

        buffer = io.BytesIO()
        ta.save(buffer, wav, tts_model.sr, format="wav")
        buffer.seek(0)


        response = make_response(buffer.read())
        response.headers['Content-Type'] = 'audio/wav'
        response.headers['Content-Disposition'] = 'attachment; filename=sound.wav'
        return response
    except Exception as e:
        logger.error(e)
        return jsonify({"message": f"tts failed", "Exception": str(e)}), 400


if __name__ == '__main__':

    app.run(host='0.0.0.0',port=5555,  debug=True)