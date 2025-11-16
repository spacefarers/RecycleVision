FROM python:3.11-slim

# Install .NET 7 runtime
RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and extract .NET 7 runtime
RUN mkdir -p /opt/dotnet && \
    cd /tmp && \
    wget -q https://dotnetcli.azureedge.net/dotnet/Runtime/7.0.20/dotnet-runtime-7.0.20-linux-arm64.tar.gz && \
    tar -xzf dotnet-runtime-7.0.20-linux-arm64.tar.gz -C /opt/dotnet && \
    rm dotnet-runtime-7.0.20-linux-arm64.tar.gz

ENV DOTNET_ROOT="/opt/dotnet" \
    PATH="/opt/dotnet:$PATH"

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

# Try to install nncase - may not be available for all platforms
RUN pip install nncase==2.10.0 || echo "Warning: nncase installation failed, may not be available for this platform"

CMD ["python", "quantize.py"]