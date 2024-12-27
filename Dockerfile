FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install git and other dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install unsloth and its dependencies
RUN pip install "unsloth[cuda] @ git+https://github.com/unslothai/unsloth.git"
RUN pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes

# Copy the rest of the application
COPY . .

# Command to run the script
CMD ["python", "copy_of_fine_tuning_llama_3_8b.py"] 