FROM walkerlab/pytorch:python3.8-torch1.9.1-cuda11.1.1-dj0.12.7

RUN pip3 install -e git+https://github.com/learning-2-learn/lfp_tools.git#egg=lfp_tools
RUN pip3 install -e git+https://github.com/learning-2-learn/spike_tools.git#egg=spike_tools

ADD . /src/wcst_decode
RUN pip3 install -e /src/wcst_decode
