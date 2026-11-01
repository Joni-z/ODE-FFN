# Upstream baseline notes

User instruction,1 November2026: copy standard repository code rather than
reimplementing a training framework. Keep the active repository simple; the
FFN should be the replaceable part. An executed kernel is not proof that a
bespoke benchmark has correct learning semantics.

Investigate both official JiT and official flow-matching image examples. A
public CIFAR result from a U-Net does not automatically validate a DiT/JiT
FFN comparison. Preserve that distinction when selecting the source.
