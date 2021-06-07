# SimCLR Experiments 
# Brightness = 0.1
# Contrast = 0.1
# Gamma = (0.5, 2.0)
# Noise = 0.1 
# Jitter prob = 0.5
# Crop factor = (0.5625, 1.0)
# Finetune decoder only = True

# 100% data
# python3 -m sss.contrastive.main --model=simclr --fraction_data=1.0 --finetune_decoder_only=True --train_epochs=50
# python3 -m sss.contrastive.main --model=simclr --fraction_data=1.0 --finetune_decoder_only=True --train_epochs=50
python3 -m sss.contrastive.main --model=simclr --fraction_data=1.0 --finetune_decoder_only=True --train_epochs=50

# 50% data
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.5 --finetune_decoder_only=True
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.5 --finetune_decoder_only=True
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.5 --finetune_decoder_only=True

# 25% data
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.25 --finetune_decoder_only=True
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.25 --finetune_decoder_only=True
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.25 --finetune_decoder_only=True

# 10% data
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.1 --finetune_decoder_only=True
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.1 --finetune_decoder_only=True
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.1 --finetune_decoder_only=True

