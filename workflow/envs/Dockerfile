

FROM continuumio/miniconda3:24.4.0-0

ENV PYTHONUNBUFFERED=1
# set PROJ lib env var for GDAL
ENV PROJ_LIB=/opt/conda/share/proj
RUN useradd -ms /bin/bash myuser
# Use bash instead of sh
SHELL ["/bin/bash", "--login", "-c"]

########################################################################
## Conda environment
########################################################################
COPY environment.yml .
RUN conda env create -n myenv -f environment.yml

########################################################################
# Install Wine for pymagicc
########################################################################
RUN dpkg --add-architecture i386 \
    && apt-get update \
    && apt-get install -y wine32:i386

########################################################################
# Create user and activate conda environment
########################################################################
USER myuser
RUN conda init
RUN echo "conda activate myenv" >> ~/.bashrc
# Test conda env 
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python", "--version"]
