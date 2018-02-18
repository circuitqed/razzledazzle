import glob
import os.path

def next_file_index(datapath,prefix=''):
    """Searches directories for files of the form *_prefix* and returns next number
        in the series"""

    dirlist=glob.glob(os.path.join(datapath,'*_'+prefix+'*'))
    dirlist.sort()
    try:
        ii=int(os.path.split(dirlist[-1])[-1].split('_')[0])+1
    except:
        ii=0
    return ii

def current_file_index(datapath,prefix=''):
    """Searches directories for files of the form *_prefix* and returns current number
        in the series"""

    dirlist=glob.glob(os.path.join(datapath,'*_'+prefix+'*'))
    dirlist.sort()
    try:
        ii=int(os.path.split(dirlist[-1])[-1].split('_')[0])
    except:
        ii=0
    return ii


def next_path_index(expt_path, prefix=''):
    dirlist = glob.glob(os.path.join(expt_path, '*' + prefix + '*[0-9][0-9][0-9]'))
    if dirlist == []:
        return 0
    dirlist.sort()
    return int(os.path.split(dirlist[-1])[-1].split('_')[-1]) + 1

def current_path_index(expt_path, prefix=''):
    dirlist = glob.glob(os.path.join(expt_path, '*' + prefix + '*[0-9][0-9][0-9]'))
    if dirlist == []:
        return 0
    dirlist.sort()
    return int(os.path.split(dirlist[-1])[-1].split('_')[-1])

def get_next_filename(datapath,prefix,suffix=''):
    ii = next_file_index(datapath, prefix)
    return "%05d_" % (ii) + prefix +suffix

def get_current_filename(datapath,prefix,suffix=''):
    ii = current_file_index(datapath, prefix)
    return "%05d_" % (ii) + prefix +suffix
