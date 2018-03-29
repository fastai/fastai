# Developed from https://github.com/Paperspace/fastai-docker/blob/master/Dockerfile
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

LABEL com.nvidia.volumes.needed="nvidia_driver"

ENV FASTAI=/fastai
ENV COURSES=${FASTAI}/courses

RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --allow-downgrades --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libnccl2=2.0.5-3+cuda9.0 \
         libnccl-dev=2.0.5-3+cuda9.0 \
         python-qt4 \
         libjpeg-dev \
	 zip \
	 unzip \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*


ENV PYTHON_VERSION=3.6
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh && \
    /opt/conda/bin/conda install conda-build

WORKDIR ${FASTAI}
COPY environment.yml ${FASTAI}
# RUN cd ${FASTAI} && ls && /opt/conda/bin/conda env create
RUN ls && /opt/conda/bin/conda env create
# RUN git clone https://github.com/fastai/fastai.git
# RUN cd fastai/ && ls && /opt/conda/bin/conda env create
RUN /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/envs/fastai/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV USER fastai

# WORKDIR ${FASTAI}
CMD source activate fastai
CMD source ~/.bashrc

WORKDIR ${COURSES}

RUN chmod -R a+w ${FASTAI} 

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]

