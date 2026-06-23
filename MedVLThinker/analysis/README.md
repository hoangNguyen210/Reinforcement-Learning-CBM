# Eval Results Analysis

Regrade results with better matching rules.

```bash
find outputs/greedy -name 'eval_results.jsonl' -exec python analysis/regrade_eval_results.py -i {} \;

find outputs/greedy -name 'eval_results.jsonl' -print0 | xargs -0 -n 1 -P 8 python analysis/regrade_eval_results.py -i
find outputs/greedy -name 'eval_results.jsonl' | parallel -j 8 python analysis/regrade_eval_results.py -i {}
```

Loss rule for llava-med which cannot follow instruction

```bash
find outputs/greedy -name 'eval_results.jsonl' -path '*llava*' -print0 | xargs -0 -n 1 -P 8 python analysis/regrade_eval_results.py --llava_med_rule -i
find outputs/greedy -name 'eval_results.jsonl' -print0 | xargs -0 -n 1 -P 8 python analysis/regrade_eval_results.py --llava_med_rule -i
```

Get the greedy results with `analysis/parse_results-v2.ipynb`.


Only sync `eval_results.jsonl`

```bash
function rsync_pull {
    local_dir=$1
    remote_dir=$2
    mkdir -p $local_dir
    mkdir -p $remote_dir
    echo $local_dir $remote_dir
    sudo rsync -avP -O --include='*/' --include='eval_results.jsonl' --exclude='*' ${remote_dir} ${local_dir}
    sudo chown $(id -u):$(id -g) -R ${local_dir}
}



local_dir=/opt/dlami/nvme/xhuan192/codes/med-vlrm/outputs/greedy/
remote_dir=~/efs/xhuan192/codes/med-vlrm/outputs/greedy/

rsync_pull $local_dir $remote_dir
```


## Eval Huatuo Vision w/ original implementation

https://github.com/xk-huang/HuatuoGPT-Vision
