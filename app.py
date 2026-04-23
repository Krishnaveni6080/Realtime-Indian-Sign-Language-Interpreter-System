from flask import Flask, render_template, Response, jsonify, request
from engine import ISLEngine
import atexit

app = Flask(__name__)
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = ISLEngine()
    return engine

@app.route('/')
def index():
    return render_template('index.html')

def gen(engine_inst):
    while True:
        frame = engine_inst.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    engine_inst = get_engine()
    return Response(gen(engine_inst),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    engine_inst = get_engine()
    data = request.json
    action = data.get('action')
    if action == 'start':
        success = engine_inst.start_camera()
        return jsonify({"status": "running" if success else "error"})
    elif action == 'stop':
        engine_inst.stop_camera()
        return jsonify({"status": "stopped"})
    return jsonify({"error": "invalid action"}), 400

@app.route('/stats')
def stats():
    engine_inst = get_engine()
    return jsonify(engine_inst.get_stats())

def cleanup():
    global engine
    if engine is not None:
        engine.release()

atexit.register(cleanup)

if __name__ == '__main__':
    get_engine()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
