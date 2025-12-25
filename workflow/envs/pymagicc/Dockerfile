

FROM continuumio/miniconda3:24.4.0-0

ARG USRNM=magicc
ARG ENVNAME=magicc
RUN useradd -ms /bin/bash ${USRNM}

########################################################################
## Conda environment
########################################################################
COPY environment.yml .
RUN conda env create -n ${ENVNAME} -f pymagicc.yml

########################################################################
# Install Wine for pymagicc
########################################################################
RUN dpkg --add-architecture i386 \
    && apt-get update \
    && apt-get install -y wine32:i386

########################################################################
# Create user and activate conda environment
########################################################################
USER ${USRNM}
RUN conda init
RUN echo "conda activate ${ENVNAME}" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]
