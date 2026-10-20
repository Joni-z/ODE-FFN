# 实验监控使用与边界

已实现：`scripts/monitor_stage.py`和`scripts/submit_stage_monitored.py`。后续冻结好新一轮代码、数据manifest和Slurm脚本后，统一通过提交器发任务；提交器自动启动独立会话中的本地监控进程，不依赖当前聊天或交互式SSH窗口。

每60秒通过SSH读取Slurm及实验目录：排队/运行状态、退出码、最新step、COMPLETE与最终checkpoint、诊断身份/配置哈希、报告complete标记。正在RUNNING但连续15分钟无新step/诊断触发stall_warning，这只是提示，长评估阶段也可能触发。失败、超时、OOM等状态与结果缺失分开记录；不自动重提或覆盖训练。最终SUCCESS要求所有任务正常退出且训练、诊断和报告齐全。完整checkpoint哈希核验仍由审计脚本执行。

本机输出：`latest.json`、`events.jsonl`、`monitor.log`、`watcher.json`；远端报告目录输出`monitoring/latest.json`与`monitoring/status.md`。SSH认证失败明确标记`MONITOR_DISCONNECTED`并重试，不把失联误报为完成。默认最多24小时，可设置`--max-hours`；到期标记`MONITOR_EXPIRED`，不宣称作业成功。

这一轮已用真实任务20378694/20378698做端到端检查，监控确认30/30训练、54/54诊断、报告完成后自动退出。测试也覆盖：作业完成但缺checkpoint、末行日志尚未写完、SSH断连、失败退出、重复提交防护，以及报告提交失败时保留训练任务号。测试没有额外提交GPU任务。

## 后续提交流程

在本机项目目录运行；以下路径是待替换模板，不可把旧stage路径当新实验重复提交：

```bash
python3 scripts/submit_stage_monitored.py \
  --runtime /scratch/zz5070/ODE_FFN/experiments/NEW_STAGE/runtime \
  --manifest configs/synthetic/NEW_STAGE/manifest.json \
  --report-root /scratch/zz5070/ODE_FFN/jit-codebase/docs/NEW_STAGE \
  --local-dir monitoring/NEW_STAGE
```

runtime中的训练/报告Slurm脚本也必须指向NEW_STAGE；当前工具面向“一数据集一数组任务”的stage协议。提交回执独占创建，网络中断后不自动重发sbatch，避免产生重复任务；需根据已保存任务号和Slurm状态核对。

```bash
tail -f monitoring/NEW_STAGE/monitor.log
```

## 边界

这是日志和状态监控，没有接入邮件、微信、Slack或其他对外消息服务，也不会自行给聊天窗口推送消息。本机需要保持开机、不休眠且能SSH到torch；网络中断期间远端训练继续，但监控无法获得新状态，恢复后继续。需要独立于本机运行的通知服务时，应另行选择明确的部署和通知渠道。

## 数据出处更正

本轮冻结快照的数据manifest注释误用了本地版本；实际训练provenance均记录远端版本。数组内容完全一致，详见data-content-audit.json。已加强主仓库的未来快照校验和提交器校验：实际执行机器的数据manifest必须与冻结清单一致。旧快照不修改，因此不能把它作为新一轮重复提交以绕过该检查。
