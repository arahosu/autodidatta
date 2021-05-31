# SimCLR Experiments 

# 100% data
python3 -m contrastive.main --model=simclr --fraction_data=1.0
python3 -m contrastive.main --model=simclr --fraction_data=1.0
python3 -m contrastive.main --model=simclr --fraction_data=1.0

# 50% data
python3 -m contrastive.main --model=simclr --fraction_data=0.5
python3 -m contrastive.main --model=simclr --fraction_data=0.5
python3 -m contrastive.main --model=simclr --fraction_data=0.5

# 10% data
python3 -m contrastive.main --model=simclr --fraction_data=0.1
python3 -m contrastive.main --model=simclr --fraction_data=0.1
python3 -m contrastive.main --model=simclr --fraction_data=0.1

# 2.5% data
python3 -m contrastive.main --model=simclr --fraction_data=0.025
python3 -m contrastive.main --model=simclr --fraction_data=0.025
python3 -m contrastive.main --model=simclr --fraction_data=0.025