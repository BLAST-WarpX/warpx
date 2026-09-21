---
name: warpx-add-picmi-parameter
description: Expose a WarpX input parameter, which C++ reads with amrex::ParmParse, in the Python PICMI interface (Python/pywarpx/picmi.py) as a typed pydantic field, e.g., `warpx_my_threshold`. Use when a C++ input parameter was added or changed and Python users need it.
argument-hint: "<input parameter, e.g., warpx.my_threshold>"
---

# Add a PICMI Parameter

Expose a WarpX input parameter in the PICMI classes of `Python/pywarpx/picmi.py`.

The procedure and its conventions (which class, field types, defaults, the `warpx_` prefix, writing the input) are documented in `Docs/source/developers/how_to_add_picmi_parameter.rst`.
**Read that file first and follow it.** The steps below add how to gather the facts and how to verify the result.

## Step 1 — Gather the facts about the C++ parameter

Take the parameter from `$ARGUMENTS` (e.g., `warpx.my_threshold` or `<species>.my_flag`), or ask the user for it.

1. Find where C++ reads it, e.g., `grep -rn '"my_threshold"' Source/`, and note:
   - the `ParmParse` prefix (e.g., `warpx`, a species name, a diagnostic name),
   - the query function: `query` / `queryWithParser` (optional), `get` / `getWithParser` (required), `queryarr` (array),
   - the C++ type, the default, and the allowed values (e.g., string choices that C++ checks).
2. Read its documentation in `Docs/source/usage/parameters.rst` (`.. pp:param:: <prefix>.<name>`): description, unit, default.
   If it is not documented, tell the user: the C++ change should document it.

## Step 2 — Find the PICMI class and the method that writes its inputs

Use the table in the how-to guide and search for other parameters of the same prefix and feature:

```bash
grep -n "pywarpx.<prefix>\." Python/pywarpx/picmi.py
```

Choose the class whose objects the parameter belongs to.
If this is ambiguous (e.g., a `warpx.*` parameter of a field solver), propose a class to the user and explain why.

## Step 3 — Add the field

Follow the how-to guide:
- type from the C++ type (`bool | None`, `int | None`, `float | None`, `Literal[...] | None`, `list[...] | None`, `Expression | None`, `float | str | None` for numbers read with the parser),
- `default=None`, so that the C++ default applies (do not copy the C++ default into Python),
- constraints (`ge`, `gt`, `le`, `lt`, `Literal`) for the values that C++ accepts,
- a `description` with unit, worded as in `parameters.rst`,
- next to related fields of the class.

Check that the class has `model_config = ConfigDict(alias_generator=warpx_options(picmistandard.PICMI_<Class>))` if it derives from a PICMI standard class, and add it if not.
Give `alias=` explicitly only if the user-facing name is not `warpx_<field name>`.

## Step 4 — Write the input parameter

In the method from step 2, assign the field to the prefix, e.g., `pywarpx.warpx.my_threshold = self.my_threshold`, next to related parameters.

## Step 5 — Verify

1. Write a small script that sets the parameter (as in the how-to guide), in the scratchpad or a temporary directory, and write the inputs without building WarpX:
   ```bash
   PYTHONPATH=Python python3 check.py && grep my_threshold inputs_check
   ```
   Check that:
   - the input line has the expected prefix, name, and value,
   - there is no line if the parameter is not given,
   - invalid values raise a `ValidationError` (if the field has constraints).
2. Run `pre-commit run --files Python/pywarpx/picmi.py`.
3. Find the PICMI tests of the feature (`Examples/**/inputs_test_*_picmi.py`), and propose to use the parameter in one of them.
   Only change tests if the user agrees. Running them needs a WarpX build with `-DWarpX_PYTHON=ON` (see `AGENTS.md`).

## Rules

- Do not rename or remove existing fields: they are the user-facing API.
- Do not edit `.pyi` stubs, `Regression/Checksum/benchmarks_json/*.json`, or `dependencies.json`.
- Do not add the parameter to `Docs/source/usage/python.rst`: the PICMI documentation is generated from the fields.

## Report

Summarize for the user: the class and field (with its user-facing name), the input line that it writes, and the output of the verification.
