# --- Stage 1: Base Image ---
FROM nvidia/cuda:12.1.1-devel-ubuntu20.04

# --- Stage 2: System & Rust Setup ---
WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_DIST_SERVER=https://mirrors.ustc.edu.cn/rust-static
ENV RUSTUP_UPDATE_ROOT=https://mirrors.ustc.edu.cn/rust-static/rustup
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# --- Stage 3: Conda Environment & Python Dependencies ---
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH /opt/conda/bin:$PATH

RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --set show_channel_urls yes && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN conda config --set report_errors false && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda create -n ph python=3.10 -y
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/conda/bin/activate ph" >> ~/.bashrc

# --- Stage 4: Install Python Dependencies ---
COPY PerceptHash/requirements.txt .
RUN source /opt/conda/bin/activate ph && \
    pip install --no-cache-dir -r requirements.txt

# --- Stage 5: Copy Code & Build ---
COPY . .

RUN source /opt/conda/bin/activate ph && \
    pip install -e PerceptHash/model/kernels/selective_scan

RUN cd HDProof && cargo build --release
# --- Stage 6: Configure Runtime ---
CMD ["/bin/bash"]