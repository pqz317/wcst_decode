FROM walkerlab/pytorch:python3.8-torch1.9.1-cuda11.1.1-dj0.12.7

RUN pip3 install git+https://github.com/learning-2-learn/lfp_tools.git

ADD . /src/wcst_decode
RUN pip install -e /src/wcst_decode
