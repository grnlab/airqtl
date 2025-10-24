FROM ubuntu:latest
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0

# Install packages
RUN apt update && \
	apt install -y sudo openssh-client nano make less wget curl python3 python3-pip && \
	rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --break-system-packages --no-cache-dir numpy scipy pandas matplotlib scikit-learn ipython jupyter notebook networkx sympy

RUN pip3 install --break-system-packages --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Set ownership of common directories to the u
RUN mkdir -p /workspace

WORKDIR /workspace

# Create vscode user with host UID/GID
ARG USER_ID=1011
ARG GROUP_ID=1011
RUN groupadd -g ${GROUP_ID} vscode && \
	useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash vscode && \
	usermod -aG sudo vscode && \
	echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Change ownership of workspace to vscode user
RUN chown -R vscode:vscode /workspace

# Switch to vscode user
USER vscode

# Default command
CMD ["/bin/bash"]



