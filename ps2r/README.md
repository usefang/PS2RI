# Sentiment-oriented Sarcasm Integration for Video Sentiment Analysis Enhancement with Sarcasm Assistance
## Pretrained Models
For visual encoder and acoustic encoder, we use the pre-trained weights in HKT. The weights file can be downloaded in https://github.com/matalvepu/HKT

## Training

You can run the below code to train PS2RI:

`python our_main.py --dataset="sarcasm" --max_seq_length=85 --batch_size=64 --learning_rate=1e-5 --device=cuda:0 --save_path=save_file_path --file_path=file_path`



