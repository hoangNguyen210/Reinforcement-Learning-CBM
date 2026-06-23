# Sync

When use `rsync` for directory, always append `/`

## Preparation
```bash
mkdir -p ~/efs
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport fs-0a1792d3c7c479e0d.efs.us-west-2.amazonaws.com:/ ~/efs

sudo mkdir -p ~/efs/"${USER}"

# Do not change own, the owner ids are different across nodes
# sudo chown -R "${USER}" ~/efs/"${USER}"
```

## Rsync

```bash
function rsync_push {
    local_dir=$1
    remote_dir=$2
    mkdir -p $local_dir
    mkdir -p $remote_dir
    sudo rsync -avP -O ${local_dir} ${remote_dir}
}

function rsync_pull {
    local_dir=$1
    remote_dir=$2
    mkdir -p $local_dir
    mkdir -p $remote_dir
    echo $local_dir $remote_dir
    sudo rsync -avP -O ${remote_dir} ${local_dir}
    sudo chown $(id -u):$(id -g) -R ${local_dir}
}


rsync_push $local_dir $remote_dir

rsync_pull $local_dir $remote_dir


# sft converted models

local_dir=/opt/dlami/nvme/xhuan192/codes/med-vlrm/outputs/sft-m23k-converted/
remote_dir=~/efs/xhuan192/codes/med-vlrm/outputs/sft-m23k-converted/

local_dir=/opt/dlami/nvme/xhuan192/codes/med-vlrm/outputs/sft-m23k/
remote_dir=~/efs/xhuan192/codes/med-vlrm/outputs/sft-m23k/

local_dir=/opt/dlami/nvme/xhuan192/codes/med-vlrm/outputs/converted/
remote_dir=~/efs/xhuan192/codes/med-vlrm/outputs/converted/

local_dir=/opt/dlami/nvme/xhuan192/codes/med-vlrm/outputs/sft-pmc_vqa-converted/
remote_dir=~/efs/xhuan192/codes/med-vlrm/outputs/sft-pmc_vqa-converted/


# misc
local_dir=/opt/dlami/nvme/xhuan192/codes/med-vlrm/misc/
remote_dir=~/efs/xhuan192/codes/med-vlrm/misc/




# checkpoints

local_dir=/opt/dlami/nvme/xhuan192/codes/med-vlrm/checkpoints/
remote_dir=~/efs/xhuan192/codes/med-vlrm/checkpoints/



# Init code and models

local_dir=/opt/dlami/nvme/xhuan192/codes/med-vlrm/data/verl/
remote_dir=~/efs/xhuan192/init_dir/med-vlrm/data/verl/

local_dir=/opt/dlami/nvme/xhuan192/codes/med-vlrm/
remote_dir=~/efs/xhuan192/init_dir/med-vlrm/


# estimate pass rate
local_dir=/opt/dlami/nvme/xhuan192/codes/med-vlrm/outputs/estimate_pass_rate/
remote_dir=~/efs/xhuan192/codes/med-vlrm/outputs/estimate_pass_rate/
```


## Sync files

Checkpoints

```bash
source_dir='/opt/dlami/nvme/xhuan192/codes/med-vlrm/checkpoints/'

target_dir=~/efs/xhuan192/codes/med-vlrm/checkpoints/
mkdir -p "${target_dir}"

tree -L 3 "$source_dir"
tree -L 3 "$target_dir"

sudo rsync -avP -O "${source_dir}" "$target_dir"
```


To avoid `rsync error: failed to set times on` on NFS, add `-O` (see https://stackoverflow.com/questions/667992/rsync-error-failed-to-set-times-on-foo-bar-operation-not-permitted).

```bash
source_dir=~/efs/xhuan192/codes/med-vlrm/checkpoints/

target_dir='/opt/dlami/nvme/xhuan192/codes/med-vlrm/checkpoints/'
mkdir -p "${target_dir}"

tree -L 3 "$source_dir"
tree -L 3 "$target_dir"

sudo rsync -avP -O "${source_dir}" "$target_dir"
```