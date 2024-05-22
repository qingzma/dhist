class Bin:
    def __init__(self) -> None:
        self.size = 0
        self.unique = 0
        self.low = None
        self.high = None

    def fit(self, data):
        pass

    def join(self, bin1: "Bin") -> int:
        pass


class NormalBin(Bin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, data):
        pass

    def join(self, bin1: "Bin") -> int:
        pass


class FinerBin(Bin):
    def __init__(self) -> None:
        super().__init__()

    def fit(self, data):
        pass

    def join(self, bin1: "Bin") -> int:
        pass
