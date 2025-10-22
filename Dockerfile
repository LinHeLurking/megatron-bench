FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:25.04-py3 AS amd64-base
ARG ARCHITECTURE=x86_64

FROM --platform=linux/arm64 nvcr.io/nvidia/pytorch:25.04-py3 AS arm64-base
ARG ARCHITECTURE=aarch64


FROM ${TARGETARCH}-base AS product
ARG TARGETARCH

# NOTE: CUDA_HOME is already correctly set by the base image
ENV CPATH=/usr/local/mpi/include:$CPATH
ENV LD_LIBRARY_PATH=/usr/local/mpi/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/lib/${ARCHITECTURE}-linux-gnu:$LD_LIBRARY_PATH

#
# Install grouped gemm
#
RUN --mount=type=cache,target=/root/.cache/pip \
    TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4

#
# Install DeepEP
#
COPY patches/deepep.patch /workspace/deepep.patch
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install nvidia-nvshmem-cu12 && \
    pushd /usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib && \
        ln -s libnvshmem_host.so.3 libnvshmem_host.so && \
    popd
RUN --mount=type=cache,target=/root/.cache/pip \
bash -ex <<"EOF"
    cd /workspace

    git clone --branch v1.2.1 --depth 1 https://github.com/deepseek-ai/DeepEP.git
    pushd DeepEP
        patch -p1 < /workspace/deepep.patch
    popd
    TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" pip install --no-build-isolation -v DeepEP/.
    rm -rf DeepEP
EOF

#
# Install Megatron
#
RUN git clone --depth 1 --branch core_r0.14.0 https://github.com/NVIDIA/Megatron-LM.git /workspace/Megatron-LM
ENV MEGATRON_PATH=/workspace/Megatron-LM
ENV PYTHONPATH="${PYTHONPATH}:${MEGATRON_PATH}"

# Install necessary dependencies
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install transformers tokenizers \
    sentencepiece modelcards datasets py-spy debugpy einops wandb

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install nvidia-mathdx
ARG TE="git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.8"
RUN --mount=type=cache,target=/root/.cache/pip \
    unset PIP_CONSTRAINT && \
    NVTE_CUDA_ARCHS="80;90;100" NVTE_BUILD_THREADS_PER_JOB=8 NVTE_FRAMEWORK=pytorch \
    pip install --no-cache-dir --no-build-isolation $TE


#
# Change the workspace
#
WORKDIR /workspace
