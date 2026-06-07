
## Open Questions / TODOs

### AMR Box Does Not Follow the Moving Window

The `warpx.fine_tag_lo` and `warpx.fine_tag_hi` parameters specify refinement regions
in **fixed lab-frame coordinates**, not in the moving window frame.

As the moving window advances, the AMR refinement box stays at its initial absolute
position and quickly leaves the region of interest (laser + wake). This causes:
- A visible rectangular artifact (resolution discontinuity boundary) in density and
  field plots after the window has moved significantly.
- The fine-level grid provides no benefit once the laser has propagated past the box.

**Question:** Is there a way to attach the AMR refinement region to the moving window?
- `warpx.fine_tag_lo/hi` are clearly lab-frame. Could a `ref_patch_function` with
  `t`-dependent expression track the window? (window position = prob_lo_z + c*t)
- Or does WarpX have a planned feature for window-following AMR?
- This might be worth asking on the WarpX GitHub Discussions:
  https://github.com/ECP-WarpX/WarpX/discussions

**Workaround used:** Run with `amr.max_level = 0` (uniform grid, no AMR) to avoid
the artifact. See `../tisa_lwfa_clean/` for the clean reference run.
