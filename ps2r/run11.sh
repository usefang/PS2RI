cross_n_head=(2 4)
fusion_dim=(192)
model_n_layers=(1 2 3 4)
sar_n_layers=(1 2 3 4)
drop_out=(0.1 0.2 0.3 0.4 0.5)
learning_rate=(1e-6 5e-6 3e-6 1e-5)

declare -i q
q=0
for i in ${!cross_n_head[@]}
do
  head=${cross_n_head[$i]}
  for j in ${!fusion_dim[@]}
  do
    dim=${fusion_dim[$j]}
    for k in ${!model_n_layers[@]}
    do
      modal_layer=${model_n_layers[$k]}
      for m in ${!sar_n_layers[@]}
      do
        sar_layer=${sar_n_layers[$k]}
        for n in ${!drop_out[@]}
        do
          drop=${drop_out[$n]}
          for u in ${!learning_rate[@]}
          do
            echo 2
            rate=${learning_rate[$u]}
            python our_main6.py --dataset="sarcasm" --max_seq_length=85 --batch_size=64 --cross_n_heads=$head --sar_fusion_dim=$dim --fusion_dim=$dim --modal_n_layers=$modal_layer --sar_n_layers=$sar_layer --weight=out4_$i$j$k$m$u --epochs=50 --dropout=$drop --learning_rate=$rate --device=cuda:1 --save_path=out5_$i$j$k$m$n$u
          done
        done
      done
    done
  done
done