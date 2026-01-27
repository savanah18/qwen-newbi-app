# Triton Multi-Stage Build

This Dockerfile supports both development and production builds using multi-stage builds.

## Build Modes

### Production Mode (default)
Installs all packages via pip during build:

```bash
# Using docker-compose
docker-compose build triton-server

# Using docker directly
cd agent/serving/triton/docker
docker build -t triton-server:prod --build-arg BUILD_MODE=prod .
```

### Development Mode
Copies site-packages from host for faster builds:

```bash
# Using docker with build context
cd agent/serving/triton/docker
docker build -t triton-server:dev \
  --build-arg BUILD_MODE=dev \
  --build-context .=/root/miniconda3/envs/aiops-py312/lib/python3.12/site-packages \
  .

# Or use the helper script
./build-dev.sh
```

### With docker-compose

```bash
# Set in .env file
BUILD_MODE=dev docker-compose build triton-server

# Or export environment variable
export BUILD_MODE=dev
docker-compose build triton-server
```

## How It Works

**Multi-Stage Architecture:**
1. **dev-stage**: Copies site-packages from build context
2. **prod-stage**: Installs packages via pip
3. **final stage**: Selects the appropriate stage based on `BUILD_MODE` argument

The `BUILD_MODE` argument determines which stage becomes the final image:
- `BUILD_MODE=dev` → Uses `dev-stage`
- `BUILD_MODE=prod` → Uses `prod-stage`

## Important Notes

- **Python Version**: Triton uses Python 3.10, not 3.12
- **Binary Compatibility**: Copying from Python 3.12 to 3.10 may cause issues
- **Recommendation**: Create a Python 3.10 conda env for dev mode compatibility
