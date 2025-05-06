import os

l_debug = False
maxlevels = 2

WarpXDIR = "../.."

os.system("grep -r DocDiagram " + WarpXDIR + "/Source > docdiagramgrep.txt")

lines = open("docdiagramgrep.txt", "r").readlines()

maindict = {}


def getnextargument(line):
    leftbracket = line.find("[")
    leftparenthesis = line.find("(")
    if leftparenthesis < 0:
        if leftbracket < 0:
            return "eol", "", ""
        else:
            rightbracket = line.find("]")
            return "[", line[leftbracket + 1 : rightbracket], line[rightbracket + 1 :]
    else:
        if 0 <= leftbracket < leftparenthesis:
            rightbracket = line.find("]")
            return "[", line[leftbracket + 1 : rightbracket], line[rightbracket + 1 :]
        else:
            rightparenthesis = line.find(")")
            return (
                "(",
                line[leftparenthesis + 1 : rightparenthesis],
                line[rightparenthesis + 1 :],
            )


i = 0
for line in lines:
    # remove prefix
    line = line[line.find("[") :]
    # get level
    level = int(line[7])
    line = line[9:]
    if l_debug:
        print("line = ", line)
    atype, anext, line = getnextargument(line)
    mydict = maindict
    while atype != "eol":
        if l_debug:
            print(atype, anext, line)
        if atype in ["[", "("]:
            if anext not in mydict:
                mydict[anext] = {}
                mydict[anext]["level"] = level
                mydict[anext]["rank"] = i
                mydict[anext]["type"] = atype
                i += 1
            mydict = mydict[anext]

        atype, anext, line = getnextargument(line)


header = """---
config:
  theme: base
  themeVariables:
    fontSize: 20px
  nodeSpacing: 20
  rankSpacing: 1
  titleTopMargin: 20
---
flowchart TD
classDef hidden display: none;
"""


def mkdiagram(dict, file, header, maxlevel=10):
    print("Generating source code...")
    f = open(file + ".mmd", "w")
    f.write(header)
    for key in dict.keys():
        bls = "   "
        f.write("subgraph " + key + "\n")
        f.write(bls + "Hidden" + key.replace(" ", "_") + '[" "]:::hidden\n')
        i = 0
        mksubdiagram(f, i, dict[key], key, bls, maxlevel)
        f.write("end\n")

    f.close()


#    print("Generating graph (png)...")
#    os.system('mmdc -i '+file+'.mmd -o '+file+'.png')
#    print("Done.")


def mksubdiagram(f, i, dict, key, bls, maxlevel):
    subkeys = dict.keys()
    parents = []
    for subkey in subkeys:
        if subkey in ["level", "rank", "type"]:
            continue
        subdict = dict[subkey]
        if subdict["level"] > maxlevel:
            continue
        identifier = key.replace(" ", "_") + "%g" % i  # subdict['rank']
        if i == 0:
            f.write(bls + "~~~" + identifier + "(" + subkey + ")\n")
            lastitem = identifier
            i += 1
        else:
            if subdict["type"] != "(":
                if parents == []:
                    f.write(bls + "-->" + identifier + "(" + subkey + ")\n")
                else:
                    for parent in parents:
                        f.write(
                            bls + parent + "-->" + identifier + "(" + subkey + ")\n"
                        )
                    parents = []
                lastitem = identifier
                i += 1
            else:
                identifier = subkey
                if identifier[:2] == "if":
                    identifier = identifier[3:]
                    if_statement = True
                    loop_statement = False
                elif identifier[:4] == "elif":
                    identifier = identifier[5:]
                    if_statement = True
                    loop_statement = False
                elif identifier[:4] == "loop":
                    identifier = identifier[5:]
                    if_statement = False
                    loop_statement = True
                else:
                    if_statement = False
                    loop_statement = False
                if if_statement:
                    f.write(bls + lastitem + "--" + identifier + "\n")
                    parents.append(
                        mksubdiagram(f, i, subdict, subkey, bls + "   ", maxlevel)
                    )
                if loop_statement:
                    loopid = "loop%g" % i
                    f.write(
                        bls
                        + lastitem
                        + "--->"
                        + loopid
                        + "([ ])"
                        + "--"
                        + identifier
                        + "\n"
                    )
                    parent = mksubdiagram(f, i, subdict, subkey, bls + "   ", maxlevel)
                    f.write(
                        bls
                        + parent
                        + "--->"
                        + "loop%ge" % i
                        + "([ ])"
                        + "--loop-->"
                        + loopid
                        + "\n"
                    )
                    f.write(bls + "loop%ge" % i)
    return lastitem


mkdiagram(maindict, "warpx", header, maxlevels)
