from pathlib import Path
from shutil import rmtree
import torch
from datetime import datetime
import os


def CleanLogs():
    """Remove old log data"""
    for path in Path("logs").glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)


def SaveModel(obj, directory, fileName):
    """Save a trained model in the specified directory"""
    path = directory + '/' + fileName
    torch.save(obj, path)


def CreateModelDir():
    """Creates a new directory for saving trained models"""
    now = datetime.now()
    date = now.strftime("%d-%m-%Y")
    time = now.strftime("%H-%M")
    dirName = date + '_' + time
    path = 'models/' + dirName
    logDir = path + '/logs'
    # create directory
    try:
        os.mkdir(path)
        os.mkdir(logDir)
    except OSError as e:
        print(e)

    return path, logDir
