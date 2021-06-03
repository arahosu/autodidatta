# Rotation Experiments 
# Finetune_decoder_only = True

# 100% data
python3 -m sss.baseline.rotation --fraction_data=1.0 --tpu=oai-tpu-2
python3 -m sss.baseline.rotation --fraction_data=1.0 --tpu=oai-tpu-2
python3 -m sss.baseline.rotation --fraction_data=1.0 --tpu=oai-tpu-2

# 50% data
python3 -m sss.baseline.rotation --fraction_data=0.5 --tpu=oai-tpu-2
python3 -m sss.baseline.rotation --fraction_data=0.5 --tpu=oai-tpu-2
python3 -m sss.baseline.rotation --fraction_data=0.5 --tpu=oai-tpu-2

# 25% data
python3 -m sss.baseline.rotation --fraction_data=0.25 --tpu=oai-tpu-2
python3 -m sss.baseline.rotation --fraction_data=0.25 --tpu=oai-tpu-2
python3 -m sss.baseline.rotation --fraction_data=0.25 --tpu=oai-tpu-2

# 10% data
python3 -m sss.baseline.rotation --fraction_data=0.1 --tpu=oai-tpu-2
python3 -m sss.baseline.rotation --fraction_data=0.1 --tpu=oai-tpu-2
python3 -m sss.baseline.rotation --fraction_data=0.1 --tpu=oai-tpu-2


