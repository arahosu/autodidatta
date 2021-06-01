# SimCLR Experiments 
# Brightness = 0.1
# Contrast = 0.1
# Gamma = (0.5, 1.0)
# Noise = 0.1 
# Jitter prob = 0.5
# Crop factor = (0.5625, 1.0)

# 100% data
python3 -m sss.contrastive.main --model=simclr --fraction_data=1.0
python3 -m sss.contrastive.main --model=simclr --fraction_data=1.0
python3 -m sss.contrastive.main --model=simclr --fraction_data=1.0

# 50% data
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.5
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.5
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.5

# 25% data
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.25
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.25
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.25

# 10% data
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.1
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.1
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.1

