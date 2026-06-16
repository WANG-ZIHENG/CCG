
#原数据集+生成青光眼
#python train_pretrain_model.py  --use_gen_data True --exclude_generated_label 0 --model_arch="convnextv2_tiny_1k_224"  --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"           --extra_train_data_dir="/root/autodl-tmp/data/1115_v_prediction_qycu_clip_loss1.1_reward_loss_uncertainty_loss_cup_disc_loss1.1_strategy_fixed_timestep_segman_ema/gen_data"
#python train_pretrain_model.py --use_gen_data True --exclude_generated_label 1 --model_arch="convnextv2_tiny_1k_224" --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"  --extra_train_data_dir="/root/autodl-tmp/data/1115_v_prediction_qycu_clip_loss1.1_reward_loss_uncertainty_loss_cup_disc_loss1.1_strategy_fixed_timestep_segman_ema/gen_data"
#
#
#python train_pretrain_model.py --use_gen_data True --exclude_generated_label 0 --model_arch="efficientnet_b0" --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k" --extra_train_data_dir="/root/autodl-tmp/data/1115_v_prediction_qycu_clip_loss1.1_reward_loss_uncertainty_loss_cup_disc_loss1.1_strategy_fixed_timestep_segman_ema/gen_data"
#python train_pretrain_model.py --use_gen_data True --exclude_generated_label 1 --model_arch="efficientnet_b0" --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k"  --extra_train_data_dir="/root/autodl-tmp/data/1115_v_prediction_qycu_clip_loss1.1_reward_loss_uncertainty_loss_cup_disc_loss1.1_strategy_fixed_timestep_segman_ema/gen_data"


#python train_pretrain_model.py --use_gen_data True --exclude_generated_label 1 --model_arch="efficientnet_b0" --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"  --extra_train_data_dir="/root/autodl-tmp/data/1115_v_prediction_qycu_clip_loss1.1_reward_loss_uncertainty_loss_cup_disc_loss1.1_strategy_fixed_timestep_segman_ema/gen_data"
#python train_pretrain_model.py --use_gen_data True --exclude_generated_label 0 --model_arch="efficientnet_b0" --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"  --extra_train_data_dir="/root/autodl-tmp/data/1115_v_prediction_qycu_clip_loss1.1_reward_loss_uncertainty_loss_cup_disc_loss1.1_strategy_fixed_timestep_segman_ema/gen_data"



# 循环执行三次
#for i in {1..1}; do
#    echo "========== Round $i =========="
##
##    # fairvlmed10k 数据集
#    python train_pretrain_model.py  --use_gen_data True --balance_attribute="gender" --model_arch="efficientnet_b0"  --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k"           --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"
#    python train_pretrain_model.py  --use_gen_data True --balance_attribute="race" --model_arch="efficientnet_b0"  --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k"          --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"
#    python train_pretrain_model.py  --use_gen_data True --balance_attribute="age" --model_arch="efficientnet_b0"  --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k"        --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"
##
#    # 10k 数据集
#    python train_pretrain_model.py  --use_gen_data True --balance_attribute="gender" --model_arch="efficientnet_b0"  --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"           --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"
#    python train_pretrain_model.py  --use_gen_data True --balance_attribute="race" --model_arch="efficientnet_b0"  --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"           --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"
#    python train_pretrain_model.py  --use_gen_data True --balance_attribute="age" --model_arch="efficientnet_b0"  --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"           --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"
##
#    echo "========== Round $i completed =========="
#    echo ""
#done
#
#python train_pretrain_model.py  --use_gen_data=True  --model_arch="efficientnet_b0"  --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k"        --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"
#python train_pretrain_model.py  --use_gen_data=True  --model_arch="efficientnet_b0"   --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"           --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"

#python train_pretrain_model.py  --use_gen_data=False  --model_arch="efficientnet_b0"  --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k"        --extra_train_data_dir=""
#python train_pretrain_model.py  --use_gen_data=False  --model_arch="efficientnet_b0"  --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"           --extra_train_data_dir=""
#
#
#
## 循环执行三次
#for i in {1..1}; do
#    echo "========== Round $i =========="
##
##    # fairvlmed10k 数据集
#    python train_pretrain_model.py  --use_gen_data True --lr=1e-7 --balance_attribute="gender" --model_arch="convnextv2_tiny_1k_224"  --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k"           --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"
#    python train_pretrain_model.py  --use_gen_data True --lr=1e-7 --balance_attribute="race" --model_arch="convnextv2_tiny_1k_224"  --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k"          --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"
#    python train_pretrain_model.py  --use_gen_data True --lr=1e-7 --balance_attribute="age" --model_arch="convnextv2_tiny_1k_224"  --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k"        --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"
##
#    # 10k 数据集
#    python train_pretrain_model.py  --use_gen_data True --lr=1e-7 --balance_attribute="gender" --model_arch="convnextv2_tiny_1k_224"  --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"           --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"
#    python train_pretrain_model.py  --use_gen_data True --lr=1e-7 --balance_attribute="race" --model_arch="convnextv2_tiny_1k_224"  --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"           --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"
#    python train_pretrain_model.py  --use_gen_data True --lr=1e-7 --balance_attribute="age" --model_arch="convnextv2_tiny_1k_224"  --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"           --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"
##
#    echo "========== Round $i completed =========="
#    echo ""
#done

#python train_pretrain_model.py  --use_gen_data=True --lr=1e-7 --model_arch="convnextv2_tiny_1k_224"  --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k"        --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"
#python train_pretrain_model.py  --use_gen_data=True --lr=1e-7 --model_arch="convnextv2_tiny_1k_224"  --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"           --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema/gen_data"

#python train_pretrain_model.py  --use_gen_data=False  --model_arch="convnextv2_tiny_1k_224"  --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k"        --extra_train_data_dir=""
#python train_pretrain_model.py  --use_gen_data=False  --model_arch="convnextv2_tiny_1k_224"  --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"           --extra_train_data_dir=""



python train_pretrain_model.py  --use_gen_data=True  --model_arch="efficientnet_b1" --cup_disc_threshold=0.7 --lr=5e-6 --samples_per_group=2 --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"           --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema_1227/gen_data"
python train_pretrain_model.py  --use_gen_data=True  --model_arch="efficientnet_b1" --cup_disc_threshold=0.7 --lr=1e-6 --samples_per_group=2 --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"           --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema_1227/gen_data"
python train_pretrain_model.py  --use_gen_data=True  --model_arch="efficientnet_b1" --cup_disc_threshold=0.7 --lr=5e-7 --samples_per_group=2 --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"           --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema_1227/gen_data"
python train_pretrain_model.py  --use_gen_data=False  --model_arch="efficientnet_b1"  --summarized_note_file="/root/autodl-tmp/data/10k/data_summary.csv" --dataset_dir="/root/autodl-tmp/data/10k"           --extra_train_data_dir=""




python train_pretrain_model.py  --use_gen_data=True  --model_arch="efficientnet_b1" --cup_disc_threshold=0.7 --lr=5e-6 --samples_per_group=2 --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k"        --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema_1227/gen_data"
python train_pretrain_model.py  --use_gen_data=True  --model_arch="efficientnet_b1" --cup_disc_threshold=0.7 --lr=1e-6 --samples_per_group=2 --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k"        --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema_1227/gen_data"
python train_pretrain_model.py  --use_gen_data=True  --model_arch="efficientnet_b1" --cup_disc_threshold=0.7 --lr=5e-7 --samples_per_group=2 --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k"        --extra_train_data_dir="/root/autodl-tmp/data/1208_v_prediction_5hf6_reward_loss_uncertainty_loss_cup_disc_loss1.1_fixed_timestep_segman_ema_1227/gen_data"
python train_pretrain_model.py  --use_gen_data=False  --model_arch="efficientnet_b1"  --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k"        --extra_train_data_dir=""


#--samples_per_group
#--cup_disc_threshold



#--balance_attribute gender | race | ethnicity | language
#baseline
#python train_pretrain_model.py  --model_arch="convnextv2_base_22k_384" --extra_train_data_dir=""


#python train_pretrain_model.py  --model_arch="convnextv2_tiny_1k_224" --summarized_note_file="/root/autodl-tmp/data/fairvlmed10k/data_summary_all.csv" --dataset_dir="/root/autodl-tmp/data/fairvlmed10k" --extra_train_data_dir="/root/autodl-tmp/data/1115_v_prediction_qycu_clip_loss1.1_reward_loss_uncertainty_loss_cup_disc_loss1.1_strategy_fixed_timestep_segman_ema/gen_data"



