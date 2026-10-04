# TEMPORARY: introduce a /RTCu run-time check failure in WarpX
p = "Source/FieldSolver/FiniteDifferenceSolver/EvolveEPML.cpp"
s = open(p).read()
anchor = "    using warpx::fields::FieldType;\n"
assert anchor in s
s = s.replace(
    anchor,
    anchor
    + """
    int tmp_uninit;
    if (patch_type == PatchType::coarse) { tmp_uninit = 1; }
    volatile int tmp_sink = tmp_uninit; // TEMPORARY: /RTCu failure on the fine patch
    amrex::ignore_unused(tmp_sink);
""",
    1,
)
open(p, "w").write(s)
print("patched", p)
