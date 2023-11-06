class JSONRepairError(Exception):
    def __init__(self, message, position):
        super().__init__(f"{message} at position {position}")
        self.position = position
