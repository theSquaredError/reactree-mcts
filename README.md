# Reactree+mcts

1. Download alfworld dataset:
`python alfworld-download.py`

Running instructions"
`docker compos up -d --build`

docker compose exec reactree bash

# inside the container
tmux new -s startx
nvidia-xconfig -a --use-display-device=None --virtual=1280x1024 --busid=PCI:2:0:0
python /workspace/alfred/scripts/startx.py 1
# Ctrl+b then d to detach
export DISPLAY=:1

cd /workspace/alfred && python scripts/check_thor.py
# expect: (300, 300, 3) / Everything works!!!
