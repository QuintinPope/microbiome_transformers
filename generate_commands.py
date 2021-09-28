base_command = "CUDA_VISIBLE_DEVICES=YYY python begin.py --samples TRAIN_DATA --sample_labels TRAIN_LABELS --val_samples VAL_DATA --val_labels VAL_LABELS --test_samples TEST_DATA --test_labels TEST_LABELS --vocab_path VOCAB_PATH --batch_size BATCH_SIZE --load_disc DISC_PATH --layers 5 --epochs EPOCHS --attn_heads 5 --cuda --log_file ~/output/different_dropout/finetune_discrim_absolute_glove_embs_full_train_comparison_dropout_DO_lr_LR_momentum_MOM_insert_RAND_INSERT_delete_RAND_DEL_AP_AUG_RUNN --freeze_opt 0 --weighted_sampler --seq_len 513 --num_labels 2 --mse --sgd --lr LR --momentum MOM --pos_embedding 'POS_EMBS' --dropout DO --random_insert RAND_INSERT --random_delete RAND_DEL --augment_probability AUG &> ~/guille/finetune_online/online_finetune_discrim_absolute_random_vocab_embs_full_train_comparison_dropout_DO_lr_LR_momentum_MOM_insert_RAND_INSERT_delete_RAND_DEL_AP_AUG_RUNN.txt"

momentum = 0.9
dropout = 0.1
insert = 0.0
delete = 0.0
lr = 0.01
augment_probability = 1.0
train_data = "~/guille/christine_novaltrain_512_otu.npy"
train_labels = "~/guille/christine_IBD_novaltrain_labels.npy"
val_data = "~/guille/christine_val_512_otu.npy"
val_labels = "~/guille/christine_IBD_val_labels.npy"
test_data = "~/guille/christine_test_512_otu.npy"
test_labels = "~/guille/christine_IBD_test_labels.npy"
vocab_path = "~/guille/vocab_embeddings.npy"
disc_path = "~/models/discsgdlr1e2/discrim/trained_glove_embeds/100epgen/5head5layer_BS_6_epoch110_disc/pytorch_model.bin"
batch_size = 8
epochs = 100
pos_embs = "absolute"

for i in range(5):
    print_command = base_command.replace("DO", str(dropout)).replace("RAND_INSERT", str(insert)).replace("RAND_DEL", str(delete))
    print_command = print_command.replace("LR", str(lr)).replace("AUG", str(augment_probability)).replace("TRAIN_DATA", train_data)
    print_command = print_command.replace("TRAIN_LABELS", train_labels).replace("VAL_DATA", val_data).replace("VAL_LABELS", val_labels)
    print_command = print_command.replace("TEST_DATA", test_data).replace("TEST_LABELS", test_labels).replace("VOCAB_PATH", vocab_path)
    print_command = print_command.replace("DISC_PATH", disc_path).replace("BATCH_SIZE", str(batch_size)).replace("POS_EMBS", pos_embs)
    print_command = print_command.replace("RUNN", str(i + 1)).replace("YYY", str(i % 4)).replace("POS_EMBS", pos_embs)
    print_command = print_command.replace("MOM", str(momentum)).replace("EPOCHS", str(epochs))
    print(str(i) + ": ")
    print(print_command)
