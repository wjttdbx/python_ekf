# 历史备份：开发已统一到 pinn-ekf

本目录自 2026-09-12 起停止日常维护。后续代码、实验和文稿修改请在
[pinn-ekf 主仓库](../pinn-ekf/README.md) 进行。

原研究入口、论文与本地数据已经迁入
[studies/python_ekf](../pinn-ekf/studies/python_ekf/README.md)，运行时共用主库的
`aerospace` 包。原控制律、闭环与指标行为有显式历史实现，迁移回归已通过。

本目录保留原始历史和本地备份，包括尚未提交的 J2 脚本及输出。
不要在这里继续维护第二份代码；完整去向与验证见
[整合记录](../pinn-ekf/docs/repository_consolidation.md)。
