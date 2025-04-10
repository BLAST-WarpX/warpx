# Overview

This explains how to generate the documentation for Warpx, and contribute to it.
More information can be found in Docs/source/developers/documentation.rst.

## Generating the documentation

### Installing the requirements

Install the Python requirements for compiling the documentation:
```
cd Docs/
python3 -m pip install -r requirements.txt
```

### Compiling the documentation

Still in the `Docs/` directory, type
```
make html
```
You can then open the file `build/html/index.html` with a standard web browser (e.g. Firefox), in order to visualize the results on your local computer.

### Cleaning the documentation

In order to remove all of the generated files, use:
```
make clean
```

### Regenerating the theory documentation

The theory is generated from Latex sources in PICSAR, and the resulting `.rst`
files are stored in the current repo, in `source/latex_theory`. In order to
regenerate the theory documentation:
```
make Latex-conversion
```
