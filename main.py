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


UPLOAD_FOLDER = '/workspace/reference_voices'
# UPLOAD_FOLDER = './'
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
    data = request.json or {}

    text = data.get("text", "")
    if not text:
        return jsonify({"message": f"tts failed"}), 400
    voice_name = data.get("voice_name", "")

    exaggeration = float(data.get("exaggeration", 0.5))
    cfg_weight = float(data.get("cfg_weight", 0.5))
    temperature = float(data.get("temperature", 0.8))

    max_new_tokens = int(data.get("max_new_tokens", 1000))
    max_cache_len = int(data.get("max_cache_len", 1500))
    repetition_penalty = float(data.get("repetition_penalty", 1.2))
    min_p = float(data.get("min_p", 0.05))
    top_p = float(data.get("top_p", 1.0))

    logger.debug(
        "text: {}, voice_name: {}, exaggeration: {}, cfg_weight: {}, temperature: {}, max_new_tokens: {}, max_cache_len: {}, repetition_penalty: {}, min_p: {}, top_p: {}"
        .format(text, voice_name, exaggeration, cfg_weight, temperature, max_new_tokens, max_cache_len, repetition_penalty, min_p, top_p))
    logger.debug("voice_name: {} UPLOAD_FOLDER: {}".format(voice_name, UPLOAD_FOLDER))

    voice_path = UPLOAD_FOLDER + "/" + voice_name
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logger.debug("preparing conditionals")
            tts_model.prepare_conditionals(voice_path)
            tts_model.conds.t3.to(dtype=dtype)
            logger.debug("conditionals prepared")
            logger.debug("generating tts")
            wav = tts_model.generate(
                text,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                max_cache_len=max_cache_len,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
        logger.debug("generated tts")
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