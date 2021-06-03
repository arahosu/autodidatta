# SimCLR Version 1

# 50% data
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.5

# 25% data
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.25
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.25
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.25
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.25
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.25

# 10$ data
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.1
python3 -m sss.contrastive.main --model=simclr --fraction_data=0.1

# SimSiam Version 1

# 100% data
python3 -m sss.contrastive.main --model=simsiam --fraction_data=1.0

# 50% data
python3 -m sss.contrastive.main --model=simsiam --fraction_data=0.5
python3 -m sss.contrastive.main --model=simsiam --fraction_data=0.5

# 25% data
python3 -m sss.contrastive.main --model=simsiam --fraction_data=0.25
python3 -m sss.contrastive.main --model=simsiam --fraction_data=0.25

# 10% data
python3 -m sss.contrastive.main --model=simsiam --fraction_data=0.1

