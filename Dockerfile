# Base image
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Set the working directory
WORKDIR /

# Update and upgrade the system packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt install -y \
    fonts-dejavu-core rsync git jq moreutils aria2 wget curl libgoogle-perftools-dev procps && \
    apt-get install -y nvidia-container-toolkit && \
    apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && apt-get clean -y

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Cleanup
RUN apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Remove the empty workspace directory, link to runpod network volume
RUN rm -rf /workspace && \
    ln -s /runpod-volume /workspace
# Copy all source (with submodules)
# COPY . ./workspace/
# RUN mv /workspace/src/* /workspace/ && rm -rf /workspace/src
# WORKDIR /workspace
# RUN echo "Downloading models..."
# RUN curl -o repositories/Fooocus/models/checkpoints/juggernautXL_v8Rundiffusion.safetensors -L https://huggingface.co/lllyasviel/fav_models/resolve/main/fav/juggernautXL_v8Rundiffusion.safetensors?download=true && echo "1/26" && \
#     curl -o repositories/Fooocus/models/checkpoints/OpenDalleV1.1.safetensors -L https://huggingface.co/dataautogpt3/OpenDalleV1.1/resolve/main/OpenDalleV1.1.safetensors && echo "Extra model" && \
#     curl -o repositories/Fooocus/models/loras/sd_xl_offset_example-lora_1.0.safetensors -L https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_offset_example-lora_1.0.safetensors?download=true && echo "2/26" && \
#     curl -o repositories/Fooocus/models/loras/sdxl_lcm_lora.safetensors -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/sdxl_lcm_lora.safetensors?download=true && echo "3/26" && \
#     curl -o repositories/Fooocus/models/inpaint/fooocus_inpaint_head.pth -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/fooocus_inpaint_head.pth?download=true && echo "4/26" && \
#     curl -o repositories/Fooocus/models/inpaint/inpaint.fooocus.patch -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/inpaint.fooocus.patch?download=true && echo "5/26" && \
#     curl -o repositories/Fooocus/models/inpaint/inpaint_v25.fooocus.patch -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/inpaint_v25.fooocus.patch?download=true && echo "6/26" && \
#     curl -o repositories/Fooocus/models/inpaint/inpaint_v26.fooocus.patch -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/inpaint_v26.fooocus.patch?download=true && echo "7/26" && \
#     curl -o repositories/Fooocus/models/controlnet/control-lora-canny-rank128.safetensors -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/control-lora-canny-rank128.safetensors?download=true && echo "8/26" && \
#     curl -o repositories/Fooocus/models/controlnet/fooocus_xl_cpds_128.safetensors -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/fooocus_xl_cpds_128.safetensors?download=true && echo "9/26" && \
#     curl -o repositories/Fooocus/models/controlnet/fooocus_ip_negative.safetensors -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/fooocus_ip_negative.safetensors?download=true && echo "10/26" && \
#     curl -o repositories/Fooocus/models/controlnet/ip-adapter-plus_sdxl_vit-h.bin -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/ip-adapter-plus_sdxl_vit-h.bin?download=true && echo "11/26" && \
#     curl -o repositories/Fooocus/models/controlnet/ip-adapter-plus-face_sdxl_vit-h.bin -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/ip-adapter-plus-face_sdxl_vit-h.bin?download=true && echo "12/26" && \
#     curl -o repositories/Fooocus/models/upscale_models/fooocus_upscaler_s409985e5.bin -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/fooocus_upscaler_s409985e5.bin?download=true && echo "13/26" && \
#     curl -o repositories/Fooocus/models/clip_vision/clip_vision_vit_h.safetensors -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/clip_vision_vit_h.safetensors?download=true && echo "14/26" && \
#     curl -o repositories/Fooocus/models/vae_approx/xlvaeapp.pth -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/xlvaeapp.pth?download=true && echo "15/26" && \
#     curl -o repositories/Fooocus/models/vae_approx/vaeapp_sd15.pth -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/vaeapp_sd15.pt?download=true && echo "16/26" && \
#     curl -o repositories/Fooocus/models/vae_approx/xl-to-v1_interposer-v3.1.safetensors -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/xl-to-v1_interposer-v3.1.safetensors?download=true && echo "17/26" && \
#     curl -o repositories/Fooocus/models/prompt_expansion/fooocus_expansion/pytorch_model.bin -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/fooocus_expansion.bin?download=true && echo "18/26" && \
#     curl -o repositories/Fooocus/models/controlnet/detection_Resnet50_Final.pth -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/detection_Resnet50_Final.pth?download=true && echo "19/26" && \
#     curl -o repositories/Fooocus/models/controlnet/detection_mobilenet0.25_Final.pth -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/detection_mobilenet0.25_Final.pth?download=true && echo "20/26" && \
#     curl -o repositories/Fooocus/models/controlnet/parsing_parsenet.pth -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/parsing_parsenet.pth?download=true && echo "21/26" && \
#     curl -o repositories/Fooocus/models/controlnet/parsing_bisenet.pth -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/parsing_bisenet.pth.pth?download=true && echo "22/26" && \
#     curl -o repositories/Fooocus/models/clip_vision/model_base_caption_capfilt_large.pth -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/model_base_caption_capfilt_large.pth?download=true && echo "23/26" && \
#     curl -o repositories/Fooocus/models/loras/sdxl_lightning_4step_lora.safetensors -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/sdxl_lightning_4step_lora.safetensors?download=true && echo "24/26" && \
#     curl -o repositories/Fooocus/models/loras/sdxl_hyper_sd_4step_lora.safetensors -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/sdxl_hyper_sd_4step_lora.safetensors?download=true && echo "25/26" && \
#     curl -o repositories/Fooocus/models/safety_checker/stable-diffusion-safety-checker.bin -L https://huggingface.co/3WaD/RunPod-Fooocus-API/resolve/main/v0.3.30/stable-diffusion-safety-checker.bin?download=true && echo "26/26"

ADD src .
RUN sed -i -e 's/\r$//' /start.sh
RUN chmod +x /start.sh
# RUN ls
# CMD ["./start.sh"]