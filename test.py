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

            # position limit (eg. 20 for starfruits)
            self.POS_LIMIT = posLimit

            # list of orderDepth objects (all the buy and sell orders)
            self.orderDepthHistory = []

            # list of cost we bought at
            self.buyingCost = []
            self.averageBuyingCost = 0

            # weighted prices for calculation
            self.priceHistory = priceHistory

            # length of liveData list
            self.PERIOD = period

            # orders to be sent
            self.orders: list[Order] = []

            # our positions for each products
            self.currentPosition = 0
            # previous position
            # what the price should be based off EMA
            self.acceptablePrice = self.getEMA()

            # orderDepth object
            self.orderDepth: OrderDepth | None = None

            # ema sma parameter
            self.SMOOTHING = smoothing

            # currently not in use yet
            self.iocOrders = 0

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

        def appendOrderDepth(self) -> None:
            """
            Add latest orderDepth object into orderDepthHistory list
            Stores "2 * period" amount of historic data
            """
            if len(self.orderDepthHistory) >= 2 * self.PERIOD:
                self.orderDepthHistory.pop(0)
            self.orderDepthHistory.append(self.orderDepth)

        def getWeightedPrice(self) -> float | None:
            """
            UNUSED
            Method of calculating average price (like getMidPrice)
            Weigh the best ask and the best bid price by their volumes, and take average
            """
            if self.getBestAskVolume() == self.getBestBidVolume() == 0:
                latestWeightedPrice = None  # to prevent us from trading
            else:
                try:
                    latestWeightedPrice = (self.getBestAskPrice() * abs(
                        self.getBestAskVolume()) + self.getBestBidPrice() * self.getBestBidVolume()) / (
                                                  self.getBestAskVolume() + self.getBestBidVolume())
                except ZeroDivisionError:
                    latestWeightedPrice = None
                    # print("Divide by:", self.getBestAskVolume(), self.getBestBidVolume())
            return latestWeightedPrice

        def getMidPrice(self) -> float | None:
            """Calculate the midpoint between best bid and best ask price"""
            # If there is nobody willing to buy or sell
            if self.getBestAskVolume() == 0 or self.getBestBidVolume() == 0:
                return None
            else:
                return (self.getBestAskVolume() + self.getBestBidPrice()) / 2

        def appendHistoricPrice(self) -> None:
            """
            Append the current price to priceHistory
            Stores "2 * period" amount of historic data
            """
            if len(self.priceHistory) >= 2 * self.PERIOD:
                self.priceHistory.pop(0)
            self.priceHistory.append(self.getMidPrice())

        # get sma (list) based on weighted prices list
        def getSMA(self, period=None) -> float:
            """
            Get the simple moving average with period = period
            If period=None (default), uses period = self.PERIOD
            Uses priceHistory (ie. average between bid and ask)
            """
            if period is None:
                period = self.PERIOD
            return pd.DataFrame(self.priceHistory).rolling(period).mean()

        def getEMA(self, period=None) -> float:
            """
            UNTESTED
            Get the exponential moving average with period = period
            If period=None (default), uses period = self.PERIOD
            Uses priceHistory (ie. average between bid and ask)
            """
            if period is None:
                period = self.PERIOD
            ema = [sum(self.priceHistory[:int(period)]) / int(period)]
            for price in self.priceHistory[int(period):]:
                ema.append((price * (self.SMOOTHING / (1 + period))) + ema[-1] * (
                        1 - (self.SMOOTHING / (1 + period))))
            return ema[-1]

        def updateAcceptablePrice(self) -> None:
            """Update acceptable price (currently based on EMA)"""
            self.acceptablePrice = self.getEMA()

        def getBollingerBands(self, std_dev_mult: float = 2) -> tuple[float, float]:
            """Get the bollinger bands based off SMA"""
            std = pd.DataFrame(self.priceHistory).rolling(self.PERIOD).std()
            sma = self.getSMA()
            print(f"SMA: {sma}")
            bollinger_up = sma + std * std_dev_mult  # Calculate top band
            bollinger_down = sma - std * std_dev_mult  # Calculate bottom band
            return bollinger_up, bollinger_down

        # max amount we can buy regardless of the price
        def getAvailableVolume(self, buy):
            """Returns how much volume is available to use assuming worst case of trades going through"""
            volume = self.currentPosition
            if buy:
                for order in self.orders:
                    volume += max(order.quantity, 0)
                # # print(f"Max buy volume for {self.PRODUCT}: {abs(self.POS_LIMIT - volume)}")
                return self.POS_LIMIT - volume
            else:
                for order in self.orders:
                    volume -= min(order.quantity, 0)
                # # print(f"Max sell volume for {self.PRODUCT}: {abs(self.POS_LIMIT + volume)}")
                return abs(self.POS_LIMIT + volume)

        # TODO What is this doing?
        # create the order to be sent out
        def createOrder(self, ioc, volume, price):
            if ioc:
                self.iocOrders += volume
                untradedVolume = volume
                # # print(f"Create order: {self.PRODUCT} is being traded at price {price} and volume {volume}")
                while untradedVolume > 0:
                    if not self.orderDepth.sell_orders:
                        # # print(f"Untraded volume when buying {self.PRODUCT}: {untradedVolume}")
                        # # print(f"Original volume: {volume}")
                        break
                    if abs(self.orderDepth.sell_orders[min(self.orderDepth.sell_orders.keys())]) <= abs(untradedVolume):
                        untradedVolume -= abs(self.orderDepth.sell_orders[min(self.orderDepth.sell_orders.keys())])
                        del self.orderDepth.sell_orders[min(self.orderDepth.sell_orders.keys())]
                    else:
                        self.orderDepth.sell_orders[min(self.orderDepth.sell_orders.keys())] += abs(untradedVolume)
                        untradedVolume = 0
                while untradedVolume < 0:
                    if not self.orderDepth.buy_orders:
                        # print(f"Untraded volume when selling {self.PRODUCT}: {untradedVolume}")
                        # # print(f"Original volume: {volume}")
                        break
                    if abs(self.orderDepth.buy_orders[max(self.orderDepth.buy_orders.keys())]) <= abs(untradedVolume):
                        untradedVolume += abs(self.orderDepth.buy_orders[max(self.orderDepth.buy_orders.keys())])
                        del self.orderDepth.buy_orders[max(self.orderDepth.buy_orders.keys())]
                    else:
                        self.orderDepth.buy_orders[max(self.orderDepth.buy_orders.keys())] -= abs(untradedVolume)
                        untradedVolume = 0
            self.orders.append(Order(self.PRODUCT, price, round(volume)))

    def __init__(self):
        # region define instrumentInfo
        self.resinInfo = self.InstrumentInfo(outer=self,
                                             product="RAINFOREST_RESIN",
                                             posLimit=50,
                                             priceHistory=[10002, 9998, 9996, 9996, 9995, 9995, 9996, 9996, 9995, 9996],
                                             period=10,
                                             smoothing=5)

        self.kelpInfo = self.InstrumentInfo(outer=self,
                                            product="KELP",
                                            posLimit=50,
                                            priceHistory=[2032, 2032, 2032, 2032, 2032, 2032, 2032, 2034, 2032, 2032],
                                            period=10,
                                            smoothing=5)

        self.inkInfo = self.InstrumentInfo(outer=self,
                                           product="SQUID_INK",
                                           posLimit=50,
                                           priceHistory=[1031, 1830, 1831, 1831, 1829, 1830, 1832, 1834, 1837, 1838],
                                           period=10,
                                           smoothing=5)

        self.allInfo = [self.resinInfo, self.kelpInfo, self.inkInfo]


    def displayOwnTrades(self, state: TradingState) -> None:
        """Print all trades that happened in the previous timestep"""
        for product in state.own_trades.keys():
            trades: list[Trade] = state.own_trades[product]
            trade: Trade
            for trade in trades:
                if trade.quantity != 0 and trade.timestamp == state.timestamp:
                    are_buyer = trade.buyer == "SUBMISSION"
                    are_seller = trade.seller == "SUBMISSION"
                    assert are_buyer != are_seller, f"are_buyer: {are_buyer}, are_seller: {are_seller}"

                    if are_buyer:
                        print("Bought", end="")
                    else:
                        print("Sold", end="")

                    print(f" {trade.quantity} {trade.symbol} at {trade.price} seashells at time={trade.timestamp}")




    # TODO figure out what this is doing
    def test_strategy(self, instrumentInfo: InstrumentInfo) -> None:
        """
        If bid price is over 10k, sell
        If ask price is under 10k, buy
        """
        if instrumentInfo.getBestAskPrice() <= 10000:
            # min of volume we should buy, market best ask volume, and volume we can buy
            volume = min(abs(instrumentInfo.getBestAskVolume()), instrumentInfo.getAvailableVolume(buy=True))

            if volume > 0:
                print("Test strategy: BUY", instrumentInfo.PRODUCT, str(volume) + "x", instrumentInfo.getBestAskPrice())
                instrumentInfo.orders.append(Order(instrumentInfo.PRODUCT, instrumentInfo.getBestAskPrice(), volume))
                instrumentInfo.currentPosition += volume
                for i in range(volume):
                    instrumentInfo.buyingCost.append(instrumentInfo.getBestAskPrice())
                instrumentInfo.averageBuyingCost = sum(instrumentInfo.buyingCost) / len(instrumentInfo.buyingCost)
                print("BUYING COST UPDATE", instrumentInfo.averageBuyingCost)

        if instrumentInfo.getBestBidPrice() >= 10000:
            volume = min(instrumentInfo.getBestBidVolume(), instrumentInfo.getAvailableVolume(buy=False))
            print(" Vol Bid ", volume)

            if volume > 0:
                print("Test strategy: SELL", str(volume) + "x", instrumentInfo.getBestBidPrice())
                instrumentInfo.orders.append(Order(instrumentInfo.PRODUCT, instrumentInfo.getBestBidPrice(), -volume))
                instrumentInfo.currentPosition -= volume
                for i in range(volume):
                    if len(instrumentInfo.buyingCost) > 0:
                        instrumentInfo.buyingCost.pop(0)

    def bollinger_band_strategy(self, instrumentInfo: InstrumentInfo, verbose=False):
        band_up, band_down = instrumentInfo.getBollingerBands(std_dev_mult=1)
        print(f"bid: {instrumentInfo.getBestBidVolume()} @ {instrumentInfo.getBestBidPrice()}")
        print(f"ask: {instrumentInfo.getBestAskVolume()} @ {instrumentInfo.getBestAskPrice()}")
        print(f"band_up: {band_up}, band_down: {band_down}")
        band_up = band_up.values[-1][0]
        band_down = band_down.values[-1][0]

        if instrumentInfo.getBestAskPrice() > band_up:
            volume = min(round(instrumentInfo.POS_LIMIT - instrumentInfo.currentPosition),
                         abs(instrumentInfo.getBestAskVolume()),
                         instrumentInfo.getAvailableVolume(buy=True))
            if volume > 0:
                instrumentInfo.orders.append(Order(instrumentInfo.PRODUCT, instrumentInfo.getBestAskPrice(), volume))
                instrumentInfo.currentPosition += volume
                print(f"Buying {volume} {instrumentInfo.PRODUCT} @ {instrumentInfo.getBestAskPrice()}")

        elif instrumentInfo.getBestAskPrice() < band_down:
            volume = min(round(instrumentInfo.POS_LIMIT - instrumentInfo.currentPosition),
                         abs(instrumentInfo.getBestAskVolume()),
                         instrumentInfo.getAvailableVolume(buy=False))
            if volume > 0:
                instrumentInfo.orders.append(Order(instrumentInfo.PRODUCT, instrumentInfo.getBestAskPrice(), -volume))
                instrumentInfo.currentPosition -= volume
                print(f"Selling {volume} {instrumentInfo.PRODUCT} @ {instrumentInfo.getBestBidPrice()}")

    def base_strategy(self, instrumentInfo: InstrumentInfo):
        band_up, band_down = instrumentInfo.getBollingerBands(std_dev_mult=1)
        band_up = band_up.values[-1][0]
        band_down = band_down.values[-1][0]

        # Check if the lowest ask (sell order) is lower than the above defined fair value
        if instrumentInfo.getBestAskPrice() < instrumentInfo.acceptablePrice and instrumentInfo.getBestAskPrice() != 0:

            possibleProfit = instrumentInfo.acceptablePrice - instrumentInfo.getBestAskPrice()
            delta = max(band_up - instrumentInfo.getBestAskPrice(), 0.1)
            # limit volume based on potential profit, less profit -> less max volume for this price
            total_volume = (instrumentInfo.POS_LIMIT / (
                    1 + np.exp((-5 * possibleProfit) / delta))) - instrumentInfo.POS_LIMIT / 2

            print(" Total Vol B ", band_up)
            # min of volume we should buy, market best ask volume, and volume we can buy
            volume = min(round(total_volume - instrumentInfo.currentPosition), abs(instrumentInfo.getBestAskVolume()),
                         instrumentInfo.getAvailableVolume(buy=True))
            print(" Vol B ", volume)
            if volume > 0:
                print("Base strategy: BUY", instrumentInfo.PRODUCT, str(volume) + "x", instrumentInfo.getBestAskPrice())
                instrumentInfo.orders.append(Order(instrumentInfo.PRODUCT, instrumentInfo.getBestAskPrice(), volume))
                instrumentInfo.currentPosition += volume
                for i in range(volume):
                    instrumentInfo.buyingCost.append(instrumentInfo.getBestAskPrice())
                instrumentInfo.averageBuyingCost = sum(instrumentInfo.buyingCost) / len(instrumentInfo.buyingCost)
                print("BUYING COST UPDATE", instrumentInfo.averageBuyingCost)

        if instrumentInfo.getBestBidPrice() > instrumentInfo.acceptablePrice and instrumentInfo.getBestBidPrice() > instrumentInfo.averageBuyingCost:
            possibleProfit = instrumentInfo.getBestBidPrice() - instrumentInfo.acceptablePrice
            delta = instrumentInfo.getBestBidPrice() - band_down
            total_volume = (instrumentInfo.POS_LIMIT / (
                    1 + np.exp((-5 * possibleProfit) / delta))) - instrumentInfo.POS_LIMIT / 2
            print(" Total Vol B ", total_volume)

            volume = min(round(instrumentInfo.currentPosition + total_volume), instrumentInfo.getBestBidVolume(),
                         instrumentInfo.getAvailableVolume(buy=False))
            print(" Vol B ", volume)

            if volume > 0:
                print("Base strategy: SELL", str(volume) + "x", instrumentInfo.getBestBidPrice())
                instrumentInfo.orders.append(Order(instrumentInfo.PRODUCT, instrumentInfo.getBestBidPrice(), -volume))
                instrumentInfo.currentPosition -= volume
                for i in range(volume):
                    if len(instrumentInfo.buyingCost) > 0:
                        instrumentInfo.buyingCost.pop(0)

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        if state.timestamp > 0:
            prevTraderData = jsonpickle.loads(state.traderData)

        # region Sets order depth and appends live/weighted prices to each instrument
        for instrumentInfo in self.allInfo:
            # Update the variables stored from last iteration
            if state.timestamp > 0:
                instrumentInfo.priceHistory = prevTraderData[instrumentInfo.PRODUCT]["priceHistory"]
                instrumentInfo.orderDepthHistory = prevTraderData[instrumentInfo.PRODUCT]["orderDepthHistory"]

            instrumentInfo.previousPosition = instrumentInfo.currentPosition
            try:
                instrumentInfo.currentPosition = state.position[instrumentInfo.PRODUCT]
            except KeyError:
                instrumentInfo.currentPosition = 0
            instrumentInfo.orderDepth = state.order_depths[instrumentInfo.PRODUCT]
            instrumentInfo.appendOrderDepth()
            instrumentInfo.appendHistoricPrice()
            # TODO Throws error in EMA
            # instrumentInfo.updateAcceptablePrice()
        # endregion

        result = {}

        self.displayOwnTrades(state)

        for product in state.order_depths.keys():

            if product == "RAINFOREST_RESIN":
                pass

            if product == "KELP":
                # self.bollinger_band_strategy(self.kelpInfo, verbose=True)
                print("priceHistory:", self.kelpInfo.priceHistory)

            if product == "SQUID_INK":
                pass


        traderData = {}
        for instrumentInfo in self.allInfo:
            result[instrumentInfo.PRODUCT] = instrumentInfo.orders
            instrumentInfo.orders = []

            traderData[instrumentInfo.PRODUCT] = {"priceHistory": instrumentInfo.priceHistory,
                                                  "orderDepthHistory": instrumentInfo.orderDepthHistory}

        conversions = 1
        return result, conversions, jsonpickle.dumps(traderData)
