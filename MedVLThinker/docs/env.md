
# Env


## Prepare code and env

```bash
git clone git@github.com:xk-huang/med-vlrm.git
cd med-vlrm

git clone git@github.com:volcengine/verl.git third_party/verl
cd third_party/verl
git checkout 54b2677
touch .env
```

`.env` file:
```
WANDB_API_KEY=?
WANDB_PROJECT=?
WANDB_MODE=?
WANDB_ENTITY=?

HF_TOKEN=?
HF_HOME=cache/
```


### Docker

```bash
docker pull whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3 
docker pull nvcr.io/nvidia/pytorch:24.12-py3
```

Start docker:
```bash
docker run -itd \
--runtime=nvidia \
--gpus all \
--net=host \
--ipc=host \
--ulimit memlock=-1 --ulimit stack=67108864 \
--cap-add=SYS_ADMIN \
-v $(pwd):$(pwd) \
-v $HOME:$HOME \
-w $(pwd) \
-e HF_HOME=$(pwd)/cache/ \
-u $(id -u):$(id -g) \
-e HOME=$HOME \
-e USER=$USER \
--memory 900g \
--name xk-verl \
whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3 \
bash
# --shm-size="100g" \

docker exec -it xk-verl bash

pip3 install -e third_party/verl[vllm]
```


## trl env

```bash
docker run -itd \
--runtime=nvidia \
--gpus all \
--net=host \
--ipc=host \
--ulimit memlock=-1 --ulimit stack=67108864 \
--cap-add=SYS_ADMIN \
-v $(pwd):$(pwd) \
-v $HOME:$HOME \
-w $(pwd) \
-e HF_HOME=$(pwd)/cache/ \
-u $(id -u):$(id -g) \
-e HOME=$HOME \
-e USER=$USER \
--name xk-trl \
nvcr.io/nvidia/pytorch:24.12-py3 \
bash
# --shm-size="100g" \

# driver 570 https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-03.html
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags

docker exec -it xk-trl bash

pip3 install trl==0.15.2 python-dotenv wandb qwen-vl-utils transformers==4.51.3
```