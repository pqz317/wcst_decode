FROM walkerlab/pytorch:python3.10-torch1.11.0-cuda11.7.0

RUN git clone https://github.com/learning-2-learn/lfp_tools /src/lfp_tools &&\
        pip3 install -e /src/lfp_tools

RUN git clone https://github.com/learning-2-learn/spike_tools.git /src/spike_tools &&\
        pip3 install -e /src/spike_tools

COPY . /src/wcst_decode
RUN pip3 install -e /src/wcst_decode