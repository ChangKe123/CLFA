export CUDA_VISIBLE_DEVICES="3"
python run_classifier.py --warmup_proportion 0.05 --attn_layer_num 3 --pooling_type 1 --num_train_epochs 30  --seed 2022 --learning_rate 1e-5 --data_dir ./datasets/sarcasm --image_dir ../dataset_image/ --output_dir ./output/bert_vit_1e-5/  --do_train --do_test --model_select VitBERT 1>exp/bert_vit2.txt 2>exp/bert_vit_err2.txt
