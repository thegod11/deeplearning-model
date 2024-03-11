LATEST_LOG_DIR=$(ls -d ./experiments/* | grep train_ncct_cta | sort -r | head -n 1)
tensorboard --logdir="${LATEST_LOG_DIR}/tb_logger" --port 9998 --bind_all