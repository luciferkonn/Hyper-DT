###
 # @Author: Jikun Kang
 # @Date: 1969-12-31 19:00:00
 # @LastEditTime: 2023-04-10 15:06:31
 # @LastEditors: Jikun Kang
 # @FilePath: /Hyper-DT/train_small_server.sh
### 

data_steps=${1}
use_wandb=${2}
samples=${3}
model_path=${4}
gw=${5}
create_hnet=${6}

echo "python train.py data_steps use_wandb samples model_path gw create_hnet"

python train.py --create_hnet=$create_hnet --max_epochs=1000 --eval_freq 100 --n_embd=512 --n_layer=4 --use_wandb=$use_wandb\
  --n_head=8 --device='cuda' --n_gpus --num_workers=10 --num_workers=10 --data_steps $data_steps --training_samples=$samples --use_gw=$gw\
  --load_path=$model_path --num_datasets 1 --use_gw=$gw --folder_prefix='/group-volume/jikun'\
  --wandb_status='offline'\
  --train_game_list 'Amidar' 'Assault' 'BankHeist' 'BattleZone' 'BeamRider' 'Breakout' 'Centipede' 'ChopperCommand' 'CrazyClimber' 'DemonAttack' 'DoubleDunk' 'FishingDerby' 'Freeway' 'Frostbite' 'Gopher' 'Gravitar' 'IceHockey'\\
  --eval_game_list 'Amidar' 'Assault' 'BankHeist' 'BattleZone' 'BeamRider' 'Breakout'
