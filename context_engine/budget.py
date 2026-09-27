class TokenBudget:
    def __init__(
        self,
        context_limit: int,
        output_reservation: int,
    ):
        self.context_limit = context_limit
        self.output_reservation = output_reservation

    @property
    def input_budget(self) -> int:
        return self.context_limit - self.output_reservation