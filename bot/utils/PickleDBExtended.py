import pickledb

def load(location, auto_dump, sig=True):
    '''Return a pickledb object. location is the path to the json file.'''
    return PickleDBExtended(location, auto_dump, sig)

class PickleDBExtended(pickledb.PickleDB):

    def set(self, key, value):
        key = str(key)

        return pickledb.PickleDB.set(self, key, value)

    def get(self, key):
        key = str(key)

        return pickledb.PickleDB.get(self, key)