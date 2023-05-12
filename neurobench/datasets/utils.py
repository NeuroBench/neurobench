"""
=====================================================================
Project:      NeuroBench
File:         utils.py
Description:  Python code describing helper functions for the dataloader
Date:         11. May 2023
=====================================================================
Copyright stuff
=====================================================================
"""
from os.path import isfile, splitext


def select_file():
    """
    Open a window, select a directory and return its path

    Returns
    ----------
    path    : str
        Path to selected directory
    """
    import tkinter
    from tkinter import filedialog

    # Suppress the Tkinter root window
    tkroot = tkinter.Tk()
    tkroot.withdraw()

    # return path to selected folder
    return str(tkinter.filedialog.askdirectory())


def valid_path(path, suffix='.mat'):
    """
    Checks if file exists and is a valid MATLAB file

    Parameters
    ----------
    path    : str
        Path to selected MATLAB file
    suffix  : str
        Suffix to allowed datafiles

    Returns
    ----------
    out     : bool
        true if valid file

    """
    # extract extension
    _, extension = splitext(path)

    # if no .mat specified, add it
    if extension == '':
        path += suffix
    # only .mat files currently supported
    elif extension != suffix:
        raise FileNotFoundError("Specified file is not a Matlab file: " + path)

    # if file does not exist, throw exception
    if not isfile(path):
        raise FileNotFoundError("No such File: " + path)

    return True
