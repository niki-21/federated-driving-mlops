# Base image
FROM python:3.9-slim

# Install dependencies and CloudWatch Agent
RUN apt-get update && apt-get install -y \
    git wget curl unzip libgl1-mesa-glx procps gnupg software-properties-common && \
    apt-get clean && \
    mkdir -p /opt/aws && \
    cd /opt/aws && \
    wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb && \
    dpkg -i amazon-cloudwatch-agent.deb && \
    rm amazon-cloudwatch-agent.deb

# Set working directory
WORKDIR /workspace

# Copy source code and CloudWatch config
COPY . .
# Use the credential-free template; inject credentials at runtime.
COPY clearml.conf.example /root/clearml.conf
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make entrypoint script executable
RUN chmod +x /workspace/entrypoint.sh

# Run entrypoint script
CMD ["/workspace/entrypoint.sh"]
