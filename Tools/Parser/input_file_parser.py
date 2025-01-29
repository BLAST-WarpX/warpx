def parse_input_file(input_file):
    """
    Parse WarpX input file.

    Parameters
    ----------
    input_file : string
        Path to input file.

    Returns
    -------
    input_dict : dictionary
        Dictionary storing WarpX input parameters
        (parameter's name stored as key, parameter's value stored as value).
    """
    input_dict = dict()
    with open(input_file) as ff:
        for line in ff:
            sline = line.split("=")
            # skip lines that are commented out, blank, or continuation of previous parameters
            skip_line = (
                sline[0].startswith("#") or sline[0].startswith("\n") or len(sline) == 1
            )
            if not skip_line:
                key = sline[0].strip()
                val = sline[1].split()
                # The value corresponding to a given key of input_dict is a list
                # of strings, from which we remove any leftover comments
                for i in range(len(val)):
                    if val[i].startswith("#"):
                        val = val[:i]
                        break
                input_dict[key] = val
    return input_dict
