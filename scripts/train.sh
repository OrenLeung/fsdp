#!/bin/bash

# On AWS, the EFA and OFI paths enable NCCL to use optimized networking.
export LD_LIBRARY_PATH=/opt/nccl/build/lib:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/aws-ofi-nccl/lib:/usr/local/cuda/lib:/usr/local/cuda/lib64:/usr/local/cuda:/usr/local/cuda/targets/x86_64-linux/lib/:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/lib:$LD_LIBRARY_PATH

MODEL_ARGS="\
--model_variant=llama3_8b
--use_dummy_dataset=True
--fsdp_activation_checkpointing=True
--selective_checkpointing=1
--sharding_strategy=hsdp
--batch_size=2
--report_interval=10
--use_torch_compile=True
--use_profiler=False
"

    # --master_addr=`scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1` \
    # --master_port="12234" \

python3 -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    main_training.py \
    ${MODEL_ARGS}

