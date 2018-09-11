FROM python:3.6-slim

RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

RUN echo 'deb http://ftp2.cn.debian.org/debian stretch main contrib non-free' >> /etc/apt/sources.list \
    && apt-get update && apt-get install -y git wget caffe-cpu

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

RUN conda install -c defaults caffe

ENTRYPOINT [ "python3" ]
