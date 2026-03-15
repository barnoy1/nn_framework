FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XAUTHORITY=/tmp/.Xauthority

WORKDIR /workspace

RUN printf 'Acquire::Retries "2";\nAcquire::ForceIPv4 "true";\nAcquire::http::Timeout "15";\nAcquire::https::Timeout "15";\n' > /etc/apt/apt.conf.d/99-network

RUN apt-get update -o Acquire::Retries=2 -o Acquire::ForceIPv4=true -o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15 \
    && apt-get install -y --no-install-recommends \
        bash \
        apt-transport-https \
        ca-certificates \
        curl \
        wget \
        gpg \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        python-is-python3 \
        xauth \
        x11-apps \
        libx11-xcb1 \
        libxcb1 \
        libxcb-dri3-0 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxfixes3 \
        libxi6 \
        libxrandr2 \
        libxrender1 \
        libxss1 \
        libxtst6 \
        libgbm1 \
        libasound2 \
        libatk1.0-0 \
        libatk-bridge2.0-0 \
        libcairo2 \
        libdrm2 \
        libgdk-pixbuf2.0-0 \
        libgl1 \
        libglib2.0-0 \
        libgtk-3-0 \
        libpango-1.0-0 \
        libsecret-1-0 \
        libnss3 \
        libxkbfile1 \
        libkrb5-3 \
        git \
        procps \
    && rm -rf /var/lib/apt/lists/*

RUN wget -qO- https://packages.microsoft.com/keys/microsoft.asc \
    | gpg --dearmor > /usr/share/keyrings/microsoft-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-archive-keyring.gpg] https://packages.microsoft.com/repos/vscode stable main" \
    > /etc/apt/sources.list.d/vscode.list \
    && apt-get update -o Acquire::Retries=2 -o Acquire::ForceIPv4=true -o Acquire::http::Timeout=15 -o Acquire::https::Timeout=15 \
    && apt-get install -y --no-install-recommends code \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR=/usr/local/bin sh \
    && chmod 0755 /usr/local/bin/uv

COPY requirements.txt /workspace/requirements.txt
RUN uv pip install --system -r /workspace/requirements.txt \
    && uv pip install --system debugpy

COPY docker/runner_entrypoint.sh /usr/local/bin/runner_entrypoint.sh
RUN chmod +x /usr/local/bin/runner_entrypoint.sh

COPY . /workspace
RUN chmod +x /workspace/docker/scripts/*.sh \
    && chmod +x /workspace/docker/scripts/utilities/*.sh

ENTRYPOINT ["/usr/local/bin/runner_entrypoint.sh"]
