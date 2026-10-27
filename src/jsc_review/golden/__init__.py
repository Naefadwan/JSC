"""Two-reviewer golden-label store used as ground truth for Phase 4 metrics.

The store is local JSON. Nothing in this package calls out to a remote
service. The sampler is deterministic (seeded) so the same findings get
promoted to candidates when the inputs are the same.
"""
