def parse_input_file(input_file):
    """
    Parse WarpX input file.

    TODO: consolidate with "read_dims_from_file" function and add support for:
    - more robust regex parsing (in read_dims_from_file)
    - included files via FILE = ... (in read_dims_from_file)
    - un-quote "" string values (in read_dims_from_file)
    - multi-line string assignments (generally to-do)

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


def input_has_value(input_dict, key, expected_value):
    """
    Check whether a key in an input dictionary has a given expected value.

    Parameters
    ----------
    input_dict : dict
        Dictionary returned by :func:`parse_input_file`.
    key : str
        Name of the input parameter to look up.
    expected_value : str
        The value to compare against the first token of the parameter.

    Returns
    -------
    bool
        ``True`` if *key* is present and its first token equals *expected_value*,
        ``False`` otherwise.
    """
    value = input_dict.get(key)
    return value is not None and value[0] == expected_value
