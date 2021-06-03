# Supervised baseline

# 100% data
python3 -m sss.contrastive.main --model=supervised --fraction_data=1.0 --custom_schedule=True --train_epochs=40
python3 -m sss.contrastive.main --model=supervised --fraction_data=1.0 --custom_schedule=True --train_epochs=40
python3 -m sss.contrastive.main --model=supervised --fraction_data=1.0 --custom_schedule=True --train_epochs=40
python3 -m sss.contrastive.main --model=supervised --fraction_data=1.0 --custom_schedule=True --train_epochs=40

# # 50% data
python3 -m sss.contrastive.main --model=supervised --fraction_data=0.5 --custom_schedule=True
python3 -m sss.contrastive.main --model=supervised --fraction_data=0.5 --custom_schedule=True
python3 -m sss.contrastive.main --model=supervised --fraction_data=0.5 --custom_schedule=True
python3 -m sss.contrastive.main --model=supervised --fraction_data=0.5 --custom_schedule=True

# # 25% data
python3 -m sss.contrastive.main --model=supervised --fraction_data=0.25 --custom_schedule=True
python3 -m sss.contrastive.main --model=supervised --fraction_data=0.25 --custom_schedule=True
python3 -m sss.contrastive.main --model=supervised --fraction_data=0.25 --custom_schedule=True
python3 -m sss.contrastive.main --model=supervised --fraction_data=0.25 --custom_schedule=True
python3 -m sss.contrastive.main --model=supervised --fraction_data=0.25 --custom_schedule=True

# 10% data
python3 -m sss.contrastive.main --model=supervised --fraction_data=0.1 --custom_schedule=True
python3 -m sss.contrastive.main --model=supervised --fraction_data=0.1 --custom_schedule=True
python3 -m sss.contrastive.main --model=supervised --fraction_data=0.1 --custom_schedule=True
python3 -m sss.contrastive.main --model=supervised --fraction_data=0.1 --custom_schedule=True



