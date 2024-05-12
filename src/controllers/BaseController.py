from helpers import get_settings

class BaseController :

    def __init__(self) :
        self.settings = get_settings()
