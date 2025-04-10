from typing import Dict, List
import pandas as pd
import numpy as np
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Trader:
    class InstrumentInfo:
        def __init__(self, outer, product, posLimit, priceHistory, period, smoothing):
            # eg. "STARFRUITS"
            self.PRODUCT = product

            # 'self' when called from Trader class (used to call methods in Trader)
            self.OUTER = outer

            # position limit (eg. 50 for kelp)
            self.POS_LIMIT = posLimit

            # previous price of this instrument
            self.priceHistory: list[float] = priceHistory

            # length of priceHistory list
            self.PERIOD = period

            # orders to be sent
            self.orders: list[Order] = []

            self.currentPosition: float = 0

            # Number of shells gained/ lost by this instrument
            self.shells: float = 0

        # region bid/ask price/volume getters
        def getBestBidPrice(self):
            if len(self.orderDepth.buy_orders) == 0:
                return 0
            return max(self.orderDepth.buy_orders.keys())

        def getBestAskPrice(self):
            if len(self.orderDepth.sell_orders) == 0:
                return 0
            return min(self.orderDepth.sell_orders.keys())

        def getBestAskVolume(self):
            return abs(self.orderDepth.sell_orders[self.getBestAskPrice()])

        def getBestBidVolume(self):
            return abs(self.orderDepth.buy_orders[self.getBestBidPrice()])

        # endregion

        # region Other utils

        def getMidPrice(self) -> float | None:
            """Calculate the midpoint between best bid and best ask price"""
            # If there is nobody willing to buy or sell
            if self.getBestAskVolume() == 0 or self.getBestBidVolume() == 0:
                return None
            else:
                return (self.getBestAskPrice() + self.getBestBidPrice()) / 2

        def appendHistoricPrice(self) -> None:
            """
            Append the current price to priceHistory
            Stores "2 * period" amount of historic data
            """
            if len(self.priceHistory) >= 2 * self.PERIOD:
                self.priceHistory.pop(0)
            self.priceHistory.append(self.getMidPrice())

        def getSMA(self, period=None) -> float | None:
            """
            Get the simple moving average with period = period
            If period=None (default), uses period = self.PERIOD
            Uses priceHistory (ie. average between bid and ask)
            """
            if period is None:
                period = self.PERIOD

            if len(self.priceHistory) < period:
                return None
            else:
                return np.mean(self.priceHistory[-period:])

        def getStdDev(self, period=None) -> float | None:
            """
            Get the standard deviation with period = period
            If period=None (default), uses period = self.PERIOD
            Uses priceHistory (ie. average between bid and ask)
            """
            if period is None:
                period = self.PERIOD

            if len(self.priceHistory) < period:
                return None
            else:
                return np.std(self.priceHistory[-period:])

        def getBollingerBands(self, std_dev_mult: float = 2) -> tuple[float, float]:
            """Get the bollinger bands based off SMA"""
            std = self.getStdDev()
            moving_average = self.getSMA()

            if std is None or moving_average is None:
                return None, None

            bollinger_up = moving_average + std * std_dev_mult  # Calculate top band
            bollinger_down = moving_average - std * std_dev_mult  # Calculate bottom band
            return bollinger_up, bollinger_down

        # endregion

    def __init__(self):
        # region define instrumentInfo
        self.resinInfo = self.InstrumentInfo(outer=self,
                                             product="RAINFOREST_RESIN",
                                             posLimit=50,
                                             priceHistory=[],
                                             period=10,
                                             smoothing=5)

        self.kelpInfo = self.InstrumentInfo(outer=self,
                                            product="KELP",
                                            posLimit=50,
                                            priceHistory=[],
                                            period=10,
                                            smoothing=5)

        self.inkInfo = self.InstrumentInfo(outer=self,
                                           product="SQUID_INK",
                                           posLimit=50,
                                           priceHistory=[],
                                           period=10,
                                           smoothing=5)

        self.allInfo = [self.resinInfo, self.kelpInfo, self.inkInfo]

        self.total_shells = 0

    # region Debug messaging functions
    def processPastTrades(self, state: TradingState, shells: dict[str, float]) -> dict[str, float]:
        """Print all trades that happened in the previous timestep"""

        for product in state.own_trades.keys():
            trades: list[Trade] = state.own_trades[product]
            trade: Trade
            for trade in trades:
                if trade.quantity != 0 and trade.timestamp == state.timestamp - 100:
                    are_buyer = trade.buyer == "SUBMISSION"
                    are_seller = trade.seller == "SUBMISSION"
                    assert are_buyer != are_seller, f"are_buyer: {are_buyer}, are_seller: {are_seller}"

                    if are_buyer:
                        print("Bought", end="")
                        shells[trade.symbol] -= trade.quantity * trade.price
                    else:
                        print("Sold", end="")
                        shells[trade.symbol] += trade.quantity * trade.price

                    print(f" {trade.quantity} {trade.symbol} at {trade.price} seashells at time={trade.timestamp}")
        return shells

    # endregion

    # region Strategies
    def bollinger_band_strategy(self, instrumentInfo: InstrumentInfo, std_dev_mult: float = 1, verbose=False):
        """Buys when bin/ ask are outside of bollinger bands"""
        band_up, band_down = instrumentInfo.getBollingerBands(std_dev_mult=std_dev_mult)
        if verbose:
            print(f"bid: {instrumentInfo.getBestBidVolume()} @ {instrumentInfo.getBestBidPrice()}")
            print(f"ask: {instrumentInfo.getBestAskVolume()} @ {instrumentInfo.getBestAskPrice()}")
            if band_up is not None and band_down is not None:
                print(f"band_up: {round(band_up)}, band_down: {round(band_down)}")
            else:
                print(f"band_up: {band_up}, band_down: {band_down}")

        if band_up is None or band_down is None:
            return

        # Sell when bid is above band_up
        if instrumentInfo.getBestBidPrice() > band_up:
            # Max amount we would be able to sell based on the position limit
            volume = instrumentInfo.POS_LIMIT - instrumentInfo.currentPosition
            if volume > 0:
                instrumentInfo.orders.append(
                    Order(instrumentInfo.PRODUCT, instrumentInfo.getBestBidPrice(), int(-volume)))
                if verbose:
                    print(f"Selling {volume} {instrumentInfo.PRODUCT} @ {instrumentInfo.getBestBidPrice()}")

        # Buy when ask is below band_down
        elif instrumentInfo.getBestAskPrice() < band_down:
            # Max amount we would be able to buy based on the position limit
            volume = instrumentInfo.POS_LIMIT - instrumentInfo.currentPosition
            if volume > 0:
                instrumentInfo.orders.append(Order(instrumentInfo.PRODUCT, instrumentInfo.getBestAskPrice(), volume))
                if verbose:
                    print(f"Buying {volume} {instrumentInfo.PRODUCT} @ {instrumentInfo.getBestAskPrice()}")

    def fair_price_strat(self, instrumentInfo: InstrumentInfo, std_devs: float = 0, verbose=False):
        fair_price = instrumentInfo.getSMA()
        std_dev = instrumentInfo.getStdDev()
        pos = instrumentInfo.currentPosition
        pos_limit = instrumentInfo.POS_LIMIT

        if fair_price is None or std_dev is None:
            return

        distance = std_dev * std_devs

        delta_fair = pos * ((3 * std_dev) / pos_limit)
        delta_fair = 0

        fair_price += delta_fair

        if verbose:
            print(f"bid: {instrumentInfo.getBestBidVolume()} @ {instrumentInfo.getBestBidPrice()}")
            print(f"ask: {instrumentInfo.getBestAskVolume()} @ {instrumentInfo.getBestAskPrice()}")
            print("Fair Price:", fair_price)

        # Sell when bid is above fair_price
        if instrumentInfo.getBestBidPrice() > fair_price + (distance / 2):
            # Max amount we would be able to sell based on the position limit
            volume = instrumentInfo.POS_LIMIT - instrumentInfo.currentPosition
            if volume > 0:
                instrumentInfo.orders.append(
                    Order(instrumentInfo.PRODUCT, instrumentInfo.getBestBidPrice(), int(-volume)))
                if verbose:
                    print(f"Selling {volume} {instrumentInfo.PRODUCT} @ {instrumentInfo.getBestBidPrice()}")

        # Buy when ask is below fair_price
        elif instrumentInfo.getBestAskPrice() < fair_price - (distance / 2):
            # Max amount we would be able to buy based on the position limit
            volume = instrumentInfo.POS_LIMIT - instrumentInfo.currentPosition
            if volume > 0:
                instrumentInfo.orders.append(Order(instrumentInfo.PRODUCT, instrumentInfo.getBestAskPrice(), volume))
                if verbose:
                    print(f"Buying {volume} {instrumentInfo.PRODUCT} @ {instrumentInfo.getBestAskPrice()}")

    # endregion

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        if state.timestamp > 0:
            prevTraderData = jsonpickle.loads(state.traderData)
            shells_dict = prevTraderData["shells"]
        else:
            shells_dict = {}
            for instrumentInfo in self.allInfo:
                shells_dict[instrumentInfo.PRODUCT] = 0

        # region Set order depth and append live/weighted prices to each instrument
        for instrumentInfo in self.allInfo:
            # Update the variables stored from last iteration
            if state.timestamp > 0:
                instrumentInfo.priceHistory = prevTraderData[instrumentInfo.PRODUCT]["priceHistory"]

            # Update instrumentInfo with currentPosition, orderDepth and HistoricPrice
            try:
                instrumentInfo.currentPosition = state.position[instrumentInfo.PRODUCT]
            except KeyError:
                instrumentInfo.currentPosition = 0
            instrumentInfo.orderDepth = state.order_depths[instrumentInfo.PRODUCT]
            instrumentInfo.appendHistoricPrice()
        # endregion

        # Process past trades and update shells
        if state.timestamp > 0:
            shells_dict = self.processPastTrades(state, shells_dict)
            self.total_shells = sum(shells_dict.values())

        # Debug messages
        print("Positions:", state.position)
        print("Shells:", shells_dict)
        print("Total Shells:", self.total_shells)

        # Perform strategies for all products
        for product in state.order_depths.keys():

            if product == "RAINFOREST_RESIN":
                self.bollinger_band_strategy(self.resinInfo, std_dev_mult=0.1, verbose=False)
                # self.fair_price_strat(self.resinInfo, std_devs=0.1, verbose=True)
                pass

            if product == "KELP":
                self.bollinger_band_strategy(self.kelpInfo, std_dev_mult=0.1, verbose=False)

            if product == "SQUID_INK":
                self.bollinger_band_strategy(self.inkInfo, std_dev_mult=0.1, verbose=False)

        traderData = {}
        result = {}
        for instrumentInfo in self.allInfo:
            # Put in the orders for all products
            result[instrumentInfo.PRODUCT] = instrumentInfo.orders
            instrumentInfo.orders = []

            # Store the variables which need to be stored for future timesteps
            traderData[instrumentInfo.PRODUCT] = {"priceHistory": instrumentInfo.priceHistory}

        traderData["shells"] = shells_dict

        conversions = 1
        return result, conversions, jsonpickle.dumps(traderData)
