gpu_num=8
cpu_num=90
# rlaunch --cpu $cpu_num --gpu $gpu_num --memory 500000 --group robotics --positive-tags A100-SXM4-80GB --max-wait-duration=24h --mount="juicefs://gwx-data:/mnt/gwx-data" -- zsh -c "source ~/.zshrc && conda activate navila && torchrun --nproc_per_node=$gpu_num /home/guowenxuan/gwx/test.py & exec zsh"

# rlaunch --cpu=12 --memory=60000 --max-wait-duration=24h --enable-sshd --preemptible no --mount="juicefs://gwx-data:/mnt/gwx-data" -- zsh

# 8 gpus
# volc ml_devinstance launch --resource_queue_id q-20260227173832-xr8pq --flavor_id ml.pni3ln.45xlarge -- zsh -c "source ~/.zshrc &&  exec zsh"

# # 1 gpus ml.pni3ln.5xlarge
# volc ml_devinstance launch --resource_queue_id q-20250728152513-cw5nw --flavor_id ml.pni3ln.5xlarge -- zsh -c "source ~/.zshrc && source /root/gwx/OneTwoVLA/.venv/bin/activate"

#2 gpus ml.pni3ln.11xlarge
# volc ml_devinstance launch --resource_queue_id q-20251219130948-j9w2k --flavor_id ml.pni3ln.11xlarge -- zsh -c "source ~/.zshrc && exec zsh"

# 4 4090 mml.xni3c.5xlarge
volc ml_devinstance launch --resource_queue_id q-20251223151605-j9psw --flavor_id ml.xni3c.11xlarge -- zsh -c "source ~/.zshrc && exec zsh"

# # 4 gpus ml.pni3ln.17xlarge
# volc ml_devinstance launch --resource_queue_id q-20251219130948-j9w2k --flavor_id ml.pni3ln.17xlarge -- zsh -c "source ~/.zshrc && exec zsh"
