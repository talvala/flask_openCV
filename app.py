from flask import Flask, request
from figures import DetectionManager

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    # payload = request.get_json()
    # return payload
    img = request.form.get('image_url')
    myfile = request.files['json_file']
    json = myfile.read()
    detection_manager = DetectionManager()
    mask = detection_manager.run(img, json)
    print mask


if __name__ == '__main__':
    app.run()
