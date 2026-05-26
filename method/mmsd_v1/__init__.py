# MMSD v1: Multimodal Self-Speculative Decoding via EAGLE-3
#
# Target model: LLaVA-1.5-13B (40 layers, hidden_dim=5120, 576 visual tokens)
# Draft model: EAGLE-3 architecture (1-layer LlamaDecoderLayeremb, fusion Linear(3D->D))
#
# Key modifications from EAGLE-3:
# 1. Hidden states from 3 configurable layers (default {2, N//2, N-3}, random during training)
# 2. Visual token compression: delta top-64 -> top-32 (random 32 during training)
# 3. Training loss: LK Loss (adaptive KL + TV blend) instead of KL
#
# See README.md for full plan.
