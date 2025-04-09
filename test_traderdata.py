from typing import Dict, List
import pandas as pd
import numpy as np
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

class Trader:
    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        if state.timestamp > 0:
            print(f"traderData: {state.traderData}")
            dict = jsonpickle.loads(state.traderData)
            print(dict)
            print(dict["var1"])
            print(f"var1: {dict['var1']}, var2: {dict['var2']}, var3: {dict['var3']}")

        var1 = "A"
        var2 = 100
        var3 = state.timestamp * 2

        traderData = jsonpickle.dumps({"var1": var1, "var2": var2, "var3": var3})
        # traderData = json.dumps(({"var1": 1}))

        return {}, 1, traderData