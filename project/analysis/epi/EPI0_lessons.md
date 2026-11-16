# EPI0 lessons

- Dropping gated C-slab with cosize-only AB → illegal address on shared-A and two-pass (job 61379150).
- Matches HYB4/HYB12: surcharge was not just unused padding for gateCache writes.
- Next: EPI0b restores surcharge to validate epilogue. If two-pass still crashes with surcharge, `copy_fragment(gC→d_frag)` is the suspect (store-only was proven in HYB14/17).
- To free smem later: find true AB extent (maybe per-stage align + cp.async overrun), or stage gate to a dedicated gmem scratch with known layout, not by shrinking below working size.
