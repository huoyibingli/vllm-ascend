# Qwen3.5 text-only cumulative tests

Stages: 01 NZ, 02 text fusion, 03 FLA GDN chunk128, 04 shared-expert
four-phase overlap, 05 weight prefetch v2, 06 MegaMoe with optional NZ
expert caches. Stage06 includes every preceding text feature.

MegaMoe retains BF16 ND expert lists; the production example also binds
NZ caches for ordinary GMM in layers 0-3 (3 GiB extra per rank). Other
layers keep ND-list GMM. All 40 NZ caches require about 30 GiB extra per
rank. Neither memory capacity nor performance has been validated on NPUs.
The prefetch controller uses independent input/output streams and GDN
lookahead. Static cache allocations are included in model memory profiling.

Extra conv shared storage, image preprocessing and ViT-only fusion are
excluded; baseline conv packing is retained. GDN uses native dispatch.
Use identical 32768 context limits, two devices, BF16 model, dependencies,
parallel settings, request workload and prefix-cache policy across cases.
Use the external production runbook for independent shell start commands.
