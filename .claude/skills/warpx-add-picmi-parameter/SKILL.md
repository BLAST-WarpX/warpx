---
name: warpx-add-picmi-parameter
description: Expose a WarpX input parameter, which C++ reads with amrex::ParmParse, in the Python PICMI interface (`Python/pywarpx/picmi.py`) as a typed pydantic field, e.g., `warpx_my_threshold`. Use when a C++ input parameter was added or changed, since Python users also need access to it.
argument-hint: "<input parameter, e.g., warpx.my_threshold>"
---

# Add a PICMI Parameter

Read `Docs/source/developers/how_to_add_picmi_parameter.rst` and follow its steps for the input parameter in `$ARGUMENTS`, or ask the user for it.

In addition:

- Before step 1, find where C++ reads the parameter (e.g., `grep -rn '"my_threshold"' Source/`) and its `.. pp:param::` entry in `Docs/source/usage/parameters.rst`, for its type, default, and allowed values.
  If it is not documented, tell the user.
- If the choice of class is ambiguous, propose one to the user and explain why.
- In step 4, write the check script outside the source tree, and change test files only if the user agrees.
- Finally, report the class, the user-facing parameter name, the input line that it writes, and the result of the check.
