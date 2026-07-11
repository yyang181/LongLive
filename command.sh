### B300
aws s3 cp s3://datamodel-code-us-west-2/yixinyang/envs/longlive2.tar.gz /local-ssd/envs/ --region us-west-2
mkdir -p /opt/conda/envs/longlive2 && tar -xzf /local-ssd/envs/longlive2.tar.gz -C /opt/conda/envs/longlive2/


# code
aws s3 cp s3://datamodel-code-us-west-2/yixinyang/id_ed25519_tencent_mac ~/.ssh/ --region us-west-2
chmod 600 ~/.ssh/id_ed25519_tencent_mac
mkdir -p /local-ssd/code && cd /local-ssd/code && git clone -c "core.sshCommand=ssh -i ~/.ssh/id_ed25519_tencent_mac -o StrictHostKeyChecking=no" git@gitee.com:yyang181/LongLive.git
cd /local-ssd/code/LongLive

# checkpoints 
/opt/conda/bin/pip install hf_xet && /opt/conda/bin/pip install hf_transfer && /opt/conda/bin/pip install -U huggingface_hub && HF_XET_HIGH_PERFORMANCE=1 /opt/conda/bin/hf download Wan-AI/Wan2.2-TI2V-5B --local-dir /local-ssd/code/LongLive/wan_models/Wan2.2-TI2V-5B

# train data
aws s3 sync s3://datamodel-code-us-west-2/yixinyang/code/LongLive/data/train /local-ssd/code/LongLive/data/train --region us-west-2

# resume checkpoints 
aws s3 sync s3://datamodel-code-us-west-2/yixinyang/code/LongLive/logs/train_dreamx_camera_i2v_b300/checkpoint_model_003000/ /local-ssd/code/LongLive/logs/train_dreamx_camera_i2v_b300/checkpoint_model_003000/ --region us-west-2


conda activate longlive2