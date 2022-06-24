FROM walkerlab/pytorch:python3.10-torch1.11.0-cuda11.7.0

RUN pip3 install -e git+https://github.com/learning-2-learn/lfp_tools.git#egg=lfp_tools
RUN pip3 install -e git+https://github.com/learning-2-learn/spike_tools.git#egg=spike_tools

ADD . /src/wcst_decode
RUN pip3 install -e /src/wcst_decode
