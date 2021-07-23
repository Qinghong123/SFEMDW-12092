""" Delete persistent memory.

Copyright (C) Seagate Technology LLC, 2021. All rights reserved.
"""
from   core.data import DataIo, datastore, STORES_SUPPORTED
from   hardware.load_manager import getHsaInfo
import pynative

# pylint:disable=input-builtin

def deleteDatastore():
    """ Interactively delete a particular data store from memory """

    cell_sn    = f'{pynative.getiniintex("identity", "mdwid"):04d}'
    hsa_sn, *_ = getHsaInfo()

    store = input(f'Delete which store? [{"|".join(STORES_SUPPORTED)}] ')
    print(store)
    confirmed = input(f'Delete {store} memory? [y] ')
    print(confirmed)

    if confirmed == 'y' and store in STORES_SUPPORTED:

        try:              datastore.delete(store)
        except Exception: print(f'Unable to delete {store} memory')
        else:             print(f'Successfully deleted {store} memory')

        DataIo.initCellStore(cell_sn)
        DataIo.initHsaStore(hsa_sn)

        print('Success')
    else:
        print('Abort')

if __name__ == '__main__':
    deleteDatastore()
