pip install fairseq/
pip install ftfy
pip install -e torchscale
pip install -e open_clip
pip install --user git+https://github.com/microsoft/DeepSpeed.git@jeffra/engine-xthru-v2
pip install tiktoken
pip install numpy==1.23.0

pip install  httpcore==0.15.0 gradio==3.9.0
pip install tensorboard
# for cornell eval
pip install opencv-python
pip install open3d
pip install imageio
pip install scikit-image

# for debug:'cv2' has no attribute '_registerMatType'
pip install "opencv-python-headless<4.3"