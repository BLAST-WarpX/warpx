#!/usr/bin/bash
cd "/home/basso9/src/WarpX/Source/Utils/"

CR="2"

OUT_DEV="cipaw_dev.out"
OUT_NEW="cipaw.out"

echo "Coarsening ratio cr=$CR" >"$OUT_DEV"
echo "Coarsening ratio cr=$CR" >"$OUT_NEW"

/bin/python3 "check_interp_points_and_weights_dev.py" <<< "$CR" >>"$OUT_DEV" 2>&1
/bin/python3 "check_interp_points_and_weights.py" <<< "$CR" >>"$OUT_NEW" 2>&1
