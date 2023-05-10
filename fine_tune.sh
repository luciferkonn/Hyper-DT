###
 # @Author: Jikun Kang
 # @Date: 2023-05-08 23:53:20
 # @LastEditTime: 2023-05-09 08:30:52
 # @LastEditors: Jikun Kang
 # @FilePath: /Hyper-DT/fine_tune.sh
### 
for i in Robotank TimePilot UpNDown VideoPinball WizardOfWor YarsRevenge Zaxxon Alien MsPacman SpaceInvaders StarGunner Pong 
do
sh add_lora.sh 20000 0 100 /home/jikun/Downloads/tf_model.pt 0 0 1 0 1 $i
sh add_lora.sh 20000 0 100 /home/jikun/Downloads/tf_model.pt 0 0 0 1 0 $i
done
