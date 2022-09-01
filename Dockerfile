# FROM walkerlab/pytorch:python3.8-torch1.11.0-cuda11.2.1
FROM walkerlab/pytorch-jupyter:cuda-11.3.1-pytorch-1.12.0-torchvision-0.12.0-torchaudio-0.11.0-ubuntu-20.04

RUN git clone https://github.com/learning-2-learn/lfp_tools /src/lfp_tools &&\
        pip3 install -e /src/lfp_tools

RUN git clone https://github.com/learning-2-learn/spike_tools.git /src/spike_tools &&\
        pip3 install -e /src/spike_tools

RUN pip install plotly==5.10.0

COPY . /src/wcst_decode
RUN pip3 install -e /src/wcst_decode
