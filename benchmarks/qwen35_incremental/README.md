# Qwen3.5 text-only cumulative tests

Stages: 01 NZ, 02 text fusion, 03 FLA GDN chunk128, 04 shared-expert
four-phase overlap, 05 weight prefetch v2. MegaMoe will be added last at stage 06. Each stage keeps the
preceding text features. Extra conv shared storage, image preprocessing
and ViT-only fusion are excluded; baseline conv packing is retained.
Use the fixed external production runbook for serving and benchmark commands.
GDN uses native dispatch without runtime patches. On supported A2/PCP1 with
FLA available, compare native FLA chunk64 with configurable FLA chunk128.
Keep the same two devices, BF16 model, dependencies, parallel settings,
request workload and prefix-cache policy for every case.
No NPU performance results are included in these branches.
