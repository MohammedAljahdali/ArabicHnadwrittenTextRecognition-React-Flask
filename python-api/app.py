from flask import Flask, request
from flask_restful import Resource, Api
from flask_cors import CORS
import numpy as np
import cv2
from inference import main
import detection
from detection_craft import Detection

app = Flask(__name__)
CORS(app)
api = Api(app)


class MyAPI(Resource):
    def get(self):
        return {'m': 'Hello World!'}
    def post(self):
        # print(request.data)
        # print('yes')
        # print(request.form['singleImage'])
        # print(request.get_json())
        # print(request.list_storage_class())
        print(request.form['singleImage'])
        filestr = request.files['image'].read()
        # convert string data to numpy array
        npimg = np.fromstring(filestr, np.uint8)
        # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        singleImage = True if request.form['singleImage'] == 'true' else False
        print(singleImage)
        if not singleImage:
            # img, line_begin_indices = detection.main(img)
            img, line_begin_indices = Detection(img)
            print(len(img))
        output = main(img)
        text_output = ''
        if not singleImage:
            for line in np.split(np.array(output['word']), line_begin_indices):
                text_output += " ".join(list(line))
                text_output += '\n'
            output['word'] = text_output
        else:
            text_output = output['word']
        res = {"filename": request.files['image'].filename, 'output': text_output}
        return res

api.add_resource(MyAPI, '/')

if __name__ == '__main__':
    app.run(debug=True)
