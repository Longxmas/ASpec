# Pinned NVIDIA CUDA 13 development image; see docs/user-guide/containers.md.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:0eee3094c71518ad31d011a594ae6ed6de72959ee07e318cb31cffe71690e90c
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv git g++ ca-certificates libibverbs1 libmlx5-1 libnuma1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/ASpec
COPY . .
RUN bash setup.sh --venv /opt/aspec-venv --python /usr/bin/python3.12
RUN bash setup.sh --target --venv /opt/aspec-target --python /usr/bin/python3.12
ENV VIRTUAL_ENV=/opt/aspec-venv
ENV PATH="/opt/aspec-venv/bin:${PATH}"
CMD ["bash"]
