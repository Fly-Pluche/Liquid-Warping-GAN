import json
from flask import Flask
from flask import request
from flask import render_template
from flask_uploads import UploadSet, IMAGES, configure_uploads, patch_request_class

import os
import torch

from Web import detect
from imitator_main import imitator

app = Flask(__name__, static_folder='static')
IMG_AND_VIDEO = tuple('avi mp4 jpg jpe jpeg png gif svg bmp'.split())

app.config["UPLOADED_PHOTOS_DEST"] = 'static/scratch'
photo = UploadSet('photos', IMG_AND_VIDEO)
configure_uploads(app, photo)
patch_request_class(app)

@torch.no_grad()
@app.route('/detector', methods=['POST'])
def detector():
    """删除上传的图片"""
    output_dir = './static/scratch'
    if os.path.exists(output_dir):
        os.system("rm -r %s" % output_dir)
        os.makedirs(output_dir)

    filename1 = photo.save(request.files['file'])  # 保存图片
    file_url = photo.url(filename1)  # 获取url
    img_path = photo.path(filename1)  # 获取存储路径

    filename3 = photo.save(request.files['file_tsf'])  # 保存图片
    tsf_url = photo.url(filename3)  # 获取url
    tsf_path = photo.path(filename3)  # 获取存储路径

    if request.form.get('更改背景') != None:
        filename2 = photo.save(request.files['back_ground'])  # 保存图片
        bg_url = photo.url(filename2)  # 获取url
        bg_path = photo.path(filename2)  # 获取存储路径
        result, tsf_gif = imitator(img_path, tsf_path, bg_path)

        filename = result.split('/')[-1]
        result = photo.url(filename)
        tsf_name = tsf_gif.split('/')[-1]
        tsf_url = photo.url(tsf_name)

        data = {'file_url': file_url, "result_url": str(result), 'tsf_url': tsf_url, 'bg_url': bg_url}  # 构造返回数据
    else:
        result, tsf_gif = imitator(img_path, tsf_path)
        filename = result.split('/')[-1]
        result = photo.url(filename)
        tsf_name = tsf_gif.split('/')[-1]
        tsf_url = photo.url(tsf_name)

        data = {'file_url': file_url, "result_url": str(result), 'tsf_url': tsf_url}  # 构造返回数据

    data = json.dumps(data)  # 转换为字符串
    return data

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    # app.run(host='127.0.0.1', port=5000)
    app.run(host='10.0.0.5', port=5002)
