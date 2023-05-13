
###
 # @Author: Jikun Kang
 # @Date: 1969-12-31 19:00:00
 # @LastEditTime: 2023-05-12 16:56:10
 # @LastEditors: Jikun Kang
 # @FilePath: /Hyper-DT/train_meta_world.sh
### 

data_steps=${1}
use_wandb=${2}
samples=${3}
model_path=${4}
gw=${5}
create_hnet=${6}
train=${7}
eval=${8}
lora=${9}
game_name=${10}

echo "data_steps" $data_steps
echo "use_wandb" $use_wandb
echo "samples" $samples
echo "model_path" $model_path
echo "gw" $gw
echo "create_hnet" $create_hnet
echo "train" $train
echo "eval" $eval
echo "lora" $lora
echo "game_name" $game_name

python train.py --create_hnet=$create_hnet --max_epochs=1000 --eval_freq 10 --n_embd=512 --n_layer=4 --use_wandb=$use_wandb\
  --n_head=8 --device='cuda' --n_gpus --num_workers=10 --data_steps $data_steps --training_samples=$samples --use_gw=$gw\
  --load_path=$model_path --num_datasets 1 --use_gw=$gw --folder_prefix='/home/jikun/Git/Hyper-DT/dataset_success' --batch_size=8 --eval=$eval --train=$train --apply_lora=$lora\
  --train_game_list 'reach-v2'\
  --eval_game_list 'Amidar'\
  --seed=123 --experiment_name='meta-world'\
  --eval_steps=1000
  # --cuda_cores='4,5,6,7'\