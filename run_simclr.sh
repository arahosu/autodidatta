# python3 -m self_supervised.TF2.models.simclr.simclr_pretrain --train_epochs=2000 && 
python3 -m self_supervised.TF2.models.simclr.simclr_finetune --linear_eval=True --weights=gs://cifar10_baseline/checkpoints/simclr_weights.ckpt &&
python3 -m self_supervised.TF2.models.simclr.simclr_finetune --linear_eval=False --weights=gs://cifar10_baseline/checkpoints/simclr_weights.ckpt 