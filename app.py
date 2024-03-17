from flask import Flask, request, jsonify
import os
from process import run
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Example route that returns a JSON response
@app.route('/voice/analyze', methods=['POST'])
def analyze():
    start_time = time.time()  # Record start time
    if 'sample' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['sample']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file and file.filename.endswith('.wav'):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        ffnn_pred, cnn_pred = run(filename,'FFNN/train-03-17-2024/FFNN-03-17-2024.pt',
            'CNN/train-03-17-2024/CNN-03-17-2024.pt')
        voiceType = 'human'
        humanprob = round(float(ffnn_pred[1]) * 100, 2)
        aiprob = round(float(ffnn_pred[0]) * 100, 2)
        if humanprob > 85:
            voiceType ='human'
        else:
            voiceType ='synthetic'
        elapsed_time = round((time.time() - start_time) * 1000, 3) # Calculate elapsed time
        os.remove(filename)
        return jsonify({"status": "success","analysis": {"detectedVoice": True,"voiceType":voiceType,
                                                         "confidenceScore": {"aiProbability": aiprob,"humanProbability": humanprob},"additionalInfo": {"emotionalTone":"neutral","backGroundNoise":"low"}},"responseTime": elapsed_time}), 200
    else:
        return jsonify({'error': 'Invalid file format'})

# Example route that accepts POST requests with JSON data
@app.route('/voice/ping', methods=['GET'])
def ping():
    return jsonify({'state':'App up and running.'})

if __name__ == '__main__':
    app.run(debug=True)

