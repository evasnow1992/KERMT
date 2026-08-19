FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
COPY environment.yml /tmp/environment.yml

# setup the web proxy for Internet access

# configure the ubuntu's mirror
RUN apt-get update
RUN apt-get install -y wget git build-essential zip unzip vim


# install Miniforge (community conda-forge installer; no Anaconda ToS)
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh \
    && /bin/bash miniforge.sh -b -p /softwares/miniforge3 \
    && rm -v miniforge.sh
ENV PATH="/softwares/miniforge3/bin:${PATH}"
ENV LD_LIBRARY_PATH="/softwares/miniforge3/lib:${LD_LIBRARY_PATH}"

# install Python packages
RUN pip install --upgrade pip

# Override CUDA detection for conda
ENV CONDA_OVERRIDE_CUDA=12.6

# update conda (base is conda-forge under Miniforge — no defaults channel, no ToS to accept)
RUN conda update -n base conda
RUN conda env create -n kermt -f /tmp/environment.yml

# clean-up
RUN rm -rf /var/lib/apt/lists/*
RUN conda clean -afy

# Note: the repo is NOT copied into the image. Agent skills and any equivalent
# host workflow bind-mount the live kermt repo checkout at /workspace (see
# agent/scripts/kermt_container.sh). Keeping the image as a pure environment
# makes rebuilds cache-friendly (only invalidates when environment.yml changes,
# not on every code edit) and avoids stale-code footguns where a baked-in /code
# and the runtime bind-mount disagree.

# Equivalent to `conda activate kermt`
SHELL ["conda", "run", "--no-capture-output", "-n", "kermt", "/bin/bash", "-c"]

# provide defaults for the executing container
CMD [ "/bin/bash" ]