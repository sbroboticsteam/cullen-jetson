"""
This module stores a variety of utility methods which serve to read configuration files
"""


def parse_cfg(cfgPath):
    """
    Reads the yolo cfg file and processes all of the blocks/layers into a list of dictionaries

    :param cfgPath: path to cfg file
    :type cfgPath: str

    :return: list of dictionaries; each dict holds information on individual layers
    :rtype: list
    """
    file = open(cfgPath, "r")
    lines = file.read().split("\n")  # Reads entire file
    lines = [x for x in lines if len(x) > 0]  # Removes empty lines
    lines = [x for x in lines if x[0] != "#"]  # Removes commented lines
    lines = [x.rstrip().lstrip() for x in lines]  # Get rid of fringe whitespace

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":  # This marks the start of a new block
            if len(block) != 0:  # If the block isn't empty. Implies repeat previous block
                blocks.append(block)  # Append the previous layer's dict to the blocks list
                block = {}  # Reset block to prepare to read in the next layer config values
            block["type"] = line[1:-1].rstrip()  # Keeps track of what type of layer we are about to read
        else:
            cfgParam, value = line.split("=")  # Split the config parameter and its values
            block[cfgParam.rstrip()] = value.lstrip()  # Append the param with its value as a new key value pair in block

    blocks.append(block)  # Here to make sure the last layer is added into blocks

    return blocks


def loadClasses(namesPath):
    """
    Reads names file and returns list of class names

    :param namesPath: Path to names file
    :type namesPath: str
    :return: List of class names
    :rtype list
    """
    fp = open(namesPath, "r")
    names = fp.read().split("\n")[:-1]
    return names


def parse_data(dataPath):
    """
    Reads a custom .data file which contains a variety of external information
    which the network needs to train and validate

    :param dataPath: Path to data file
    :return: Returns a dictionary of parameters within data file
    :rtype dict
    """

    options = dict()

    with open(dataPath, "r") as dp:
        lines = dp.readlines()

    for line in lines:
        line = line.strip()
        if line == "" or line.startswith("#"):
            continue

        param, value = line.split("=")
        options[param.strip()] = value.strip()

    return options
