name=op-hmdb51-dg-p3d-resnet50
rm -rf output

python3 -m torch.distributed.launch --nproc_per_node=4 train_3d.py --config_file=record/$name.yml

python3 -m torch.distributed.launch --nproc_per_node=4 extract_score_3d.py --config_file=record/$name.yml --crop_idx=0

python3 -m torch.distributed.launch --nproc_per_node=4 extract_score_3d.py --config_file=record/$name.yml --crop_idx=1

python3 -m torch.distributed.launch --nproc_per_node=4 extract_score_3d.py --config_file=record/$name.yml --crop_idx=2

python3 merge_score.py --config_file=record/$name.yml > record/$name.result.txt
