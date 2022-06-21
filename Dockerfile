FROM walkerlab/pytorch

RUN pip3 install datajoint

RUN pip3 install --upgrade git+https://github.com/learning-2-learn/lfp_tools.git

ADD . /src/wcst-decode