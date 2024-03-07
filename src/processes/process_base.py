class Process(object):
    """Process base class
    """
    name = ""

    def run(self):
        raise NotImplementedError
