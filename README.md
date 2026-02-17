# jit-codebase
Accelerator+wandb implementation of JiT

## Wandb
- 首次使用在终端执行一次：`wandb login`（按提示输入 API key，或从 https://wandb.ai/authorize 获取）。
- 训练时会自动上报：loss、学习率、FID、以及采样图。项目名和 run 名由各 config 里 `logging.project` / `logging.run_name` 决定。
- 若要把 run 记到你的个人账号或某个 team 下，在 config 的 `logging` 下取消注释并填写 `entity: "你的wandb用户名或team名"`。
