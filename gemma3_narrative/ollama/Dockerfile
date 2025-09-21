    # Base image from NVIDIA, guaranteed to have correct CUDA drivers and binaries
    FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

    # Set non-interactive mode for apt-get to avoid prompts
    ENV DEBIAN_FRONTEND=noninteractive

    # Install dependencies for Ollama (curl, git) and common tools (jq)
    RUN apt-get update && apt-get install -y \
        curl \
        git \
        jq \
        # Clean up APT cache to reduce image size
        && rm -rf /var/lib/apt/lists/*

    # Install Ollama from the official installer script
    RUN curl -fsSL https://ollama.com/install.sh | sh

    # Copy the startup script into the container
    COPY ollama_startup.sh /usr/local/bin/ollama_startup.sh

    # Give the script executable permissions inside the container as the root user
    RUN chmod +x /usr/local/bin/ollama_startup.sh

    # Set a non-root user (optional but good practice)
    RUN useradd -ms /bin/bash ollama
    USER ollama
    WORKDIR /home/ollama

    # Set the entrypoint to a script that will manage startup and model pulls
    ENTRYPOINT ["/usr/local/bin/ollama_startup.sh"]
