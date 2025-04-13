from typing import Dict, List
import pandas as pd
import numpy as np
import jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

LARGE_NUMBER = 10 ** 6
class Trader:
    class InstrumentInfo:
        def __init__(self, outer, product, posLimit, priceHistory, period, smoothing):
            # eg. "STARFRUITS"
            self.otherStoredVars = {}
            self.otherStoredVarsPrev = {}
            self.orderDepth = None
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
                return LARGE_NUMBER
            return min(self.orderDepth.sell_orders.keys())

        def getBestAskVolume(self):
            if len(self.orderDepth.sell_orders) == 0:
                return 0
            return abs(self.orderDepth.sell_orders[self.getBestAskPrice()])

        def getBestBidVolume(self):
            if len(self.orderDepth.sell_orders) == 0:
                return 0
            return abs(self.orderDepth.buy_orders[self.getBestBidPrice()])

        # endregion

        def setOrder(self, price: int, volume: int, side: str, verbose: bool = False) -> None:
            """
            Create a buy or sell order for this instrument
            :param price: Price to buy/ sell at
            :param volume: Amount to buy/ sell
            :param side: Whether to 'buy' or 'sell'
            :param verbose: Whether to print debug information
            """
            if side == "buy":
                if volume > self.POS_LIMIT - self.currentPosition:
                    print(f"Volume {volume} + current position {self.currentPosition} was above "
                          f"position limit {self.POS_LIMIT}. Decreasing volume to {self.POS_LIMIT - self.currentPosition}")
                    volume = self.POS_LIMIT - self.currentPosition

                if verbose:
                    print(f"Buying {volume} {self.PRODUCT} @ {price}")

                self.orders.append(Order(self.PRODUCT, price, int(volume)))

                self.currentPosition += volume

                remaining_volume = volume
                # Buy from existing ASKS (sell orders) at or below your price
                asks = sorted(self.orderDepth.sell_orders.keys())  # Best ask first
                for ask_price in asks:
                    if ask_price <= price:
                        available_volume = self.orderDepth.sell_orders[ask_price]
                        trade_volume = min(available_volume, remaining_volume)

                        # Update the order book
                        self.orderDepth.sell_orders[ask_price] -= trade_volume
                        if self.orderDepth.sell_orders[ask_price] == 0:
                            del self.orderDepth.sell_orders[ask_price]

                        remaining_volume -= trade_volume
                        if remaining_volume == 0:
                            break

            elif side == "sell":
                if volume > self.POS_LIMIT + self.currentPosition:
                    print(f"- Volume {volume} + current position {self.currentPosition} was below "
                          f"position limit {self.POS_LIMIT}. Decreasing volume to {self.POS_LIMIT + self.currentPosition}")
                    volume = self.POS_LIMIT + self.currentPosition

                if verbose:
                    print(f"Selling {volume} {self.PRODUCT} @ {price}")

                self.orders.append(Order(self.PRODUCT, price, int(-volume)))

                self.currentPosition -= volume

                remaining_volume = volume
                # Sell into existing BIDS (buy orders) at or above your price
                bids = sorted(self.orderDepth.buy_orders.keys(), reverse=True)  # Best bid first
                for bid_price in bids:
                    if bid_price >= price:
                        available_volume = self.orderDepth.buy_orders[bid_price]
                        trade_volume = min(available_volume, remaining_volume)

                        # Update the order book
                        self.orderDepth.buy_orders[bid_price] -= trade_volume
                        if self.orderDepth.buy_orders[bid_price] == 0:
                            del self.orderDepth.buy_orders[bid_price]

                        remaining_volume -= trade_volume
                        if remaining_volume == 0:
                            break

            else:
                raise ValueError("side must be 'buy' or 'sell'.")


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

        def getEMA(self, period=None) -> float | None:
            """
            Get the exponential moving average with period = period.
            If period=None (default), uses period = self.PERIOD.
            Uses priceHistory (average between bid and ask).
            """
            if period is None:
                period = self.PERIOD

            if len(self.priceHistory) < period:
                return None

            # Use the last 'period' prices for EMA calculation (like SMA's window)
            prices = self.priceHistory[-period:]
            smoothing_factor = 2 / (period + 1)  # Standard EMA smoothing

            # Initialize EMA with the oldest price in the window
            ema = prices[0]

            # Iterate through the remaining prices in the window
            for price in prices[1:]:
                ema = (price * smoothing_factor) + (ema * (1 - smoothing_factor))

            return float(ema)

        def getLR(self, y_list):
            x = np.arange(len(y_list))
            y = np.array(y_list)

            mean_x = np.mean(x)
            mean_y = np.mean(y)

            numerator = np.sum((x - mean_x) * (y - mean_y))
            denominator = np.sum((x - mean_x) ** 2)

            if denominator == 0:
                return mean_y  # Handles cases with insufficient data points

            m = numerator / denominator
            b = mean_y - m * mean_x

            next_x = len(y_list)
            next_y = m * next_x + b

            return next_y

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
                                             period=15,
                                             smoothing=5)

        self.kelpInfo = self.InstrumentInfo(outer=self,
                                            product="KELP",
                                            posLimit=50,
                                            priceHistory=[],
                                            period=6,
                                            smoothing=5)

        self.inkInfo = self.InstrumentInfo(outer=self,
                                           product="SQUID_INK",
                                           posLimit=50,
                                           priceHistory=[],
                                           period=10,
                                           smoothing=5)

        self.basket1Info = self.InstrumentInfo(outer=self,
                                               product="PICNIC_BASKET1",
                                               posLimit=60,
                                               priceHistory=[],
                                               period=1,
                                               smoothing=5)

        self.basket2Info = self.InstrumentInfo(outer=self,
                                               product="PICNIC_BASKET2",
                                               posLimit=100,
                                               priceHistory=[],
                                               period=15,
                                               smoothing=5)

        self.croissantInfo = self.InstrumentInfo(outer=self,
                                                 product="CROISSANTS",
                                                 posLimit=250,
                                                 priceHistory=[],
                                                 period=10,
                                                 smoothing=5)

        self.jamInfo = self.InstrumentInfo(outer=self,
                                           product="JAMS",
                                           posLimit=350,
                                           priceHistory=[],
                                           period=10,
                                           smoothing=5)

        self.djembeInfo = self.InstrumentInfo(outer=self,
                                              product="DJEMBES",
                                              posLimit=60,
                                              priceHistory=[],
                                              period=10,
                                              smoothing=5)

        self.allInfo = [self.resinInfo, self.kelpInfo, self.inkInfo,
                        self.basket1Info, self.basket2Info, self.croissantInfo, self.jamInfo, self.djembeInfo]

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
        """Buys when bin/ ask are outside bollinger bands"""
        band_up, band_down = instrumentInfo.getBollingerBands(std_dev_mult=std_dev_mult)
        if verbose:
            print("bollinger_band_strategy")
            print("Instrument:", instrumentInfo.PRODUCT)
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
            volume = instrumentInfo.POS_LIMIT + instrumentInfo.currentPosition
            if volume > 0:
                # instrumentInfo.orders.append(Order(instrumentInfo.PRODUCT, instrumentInfo.getBestBidPrice(), int(-volume)))
                instrumentInfo.setOrder(instrumentInfo.getBestBidPrice(), int(volume), "sell")
                if verbose:
                    print(f"Selling {volume} {instrumentInfo.PRODUCT} @ {instrumentInfo.getBestBidPrice()}")

        # Buy when ask is below band_down
        elif instrumentInfo.getBestAskPrice() < band_down:
            # Max amount we would be able to buy based on the position limit
            volume = instrumentInfo.POS_LIMIT - instrumentInfo.currentPosition
            if volume > 0:
                # instrumentInfo.orders.append(Order(instrumentInfo.PRODUCT, instrumentInfo.getBestAskPrice(), volume))
                instrumentInfo.setOrder(instrumentInfo.getBestAskPrice(), int(volume), "buy")
                if verbose:
                    print(f"Buying {volume} {instrumentInfo.PRODUCT} @ {instrumentInfo.getBestAskPrice()}")

    def fair_price_strat(self, instrumentInfo: InstrumentInfo, std_devs: float = 0, always_transact: float = 1.5,
                         known_price=None, verbose=False):
        # default_fair_price = instrumentInfo.getLR(instrumentInfo.priceHistory[-instrumentInfo.PERIOD:])
        default_fair_price = instrumentInfo.getSMA()
        std_dev = instrumentInfo.getStdDev()
        pos = instrumentInfo.currentPosition
        pos_limit = instrumentInfo.POS_LIMIT

        if default_fair_price is None or std_dev is None:
            return

        distance = std_dev * std_devs
        always_transact_value = (always_transact - std_devs) * std_dev

        delta_fair = -pos * (always_transact_value / pos_limit)

        fair_price = default_fair_price + delta_fair

        if known_price is not None:
            fair_price = known_price

        if verbose:
            print("fair_price_strategy")
            print("Instrument:", instrumentInfo.PRODUCT)
            print(f"bid: {instrumentInfo.getBestBidVolume()} @ {instrumentInfo.getBestBidPrice()}")
            print(f"ask: {instrumentInfo.getBestAskVolume()} @ {instrumentInfo.getBestAskPrice()}")
            print("sma:", default_fair_price)
            print("Fair Price:", fair_price)
            print("Sell at: ", fair_price + (distance / 2))
            print("Buy at: ", fair_price - (distance / 2))

        # Sell when bid is above fair_price
        if instrumentInfo.getBestBidPrice() >= fair_price + (distance / 2):
            # Max amount we would be able to sell based on the position limit
            volume = instrumentInfo.POS_LIMIT + instrumentInfo.currentPosition
            if volume > 0:
                # price = instrumentInfo.getBestBidPrice()
                price = fair_price + (distance / 2)
                instrumentInfo.setOrder(int(price), int(volume), "sell", verbose=True)
                # instrumentInfo.orders.append(Order(instrumentInfo.PRODUCT, int(price), int(-volume)))
                if verbose:
                    print(f"Selling {volume} {instrumentInfo.PRODUCT} @ {price}")

        # Buy when ask is below fair_price
        elif instrumentInfo.getBestAskPrice() <= fair_price - (distance / 2):
            # Max amount we would be able to buy based on the position limit
            volume = instrumentInfo.POS_LIMIT - instrumentInfo.currentPosition
            if volume > 0:
                # price = instrumentInfo.getBestAskPrice()
                price = fair_price - (distance / 2)
                instrumentInfo.setOrder(int(price), int(volume), "buy", verbose=True)
                # instrumentInfo.orders.append(Order(instrumentInfo.PRODUCT, int(price), int(volume)))
                if verbose:
                    print(f"Buying {volume} {instrumentInfo.PRODUCT} @ {price}")

    def basket_strategy(self, basket: InstrumentInfo, components: list[InstrumentInfo], amounts: list[int], is_premium: bool, verbose=True):
        """
        Market-neutral strategy (always keeps #basket = - #components in the basket)
        If you can buy the components for less than you can sell the basket for, buy the components and sell the basket
        and vice versa
        :param basket: The basket info
        :param components: Component instrument info
        :param amounts: Number of the corresponding component in the basket
        :param is_premium: If true, there is a premium (can be negative) for owning a basket. Takes average premium over PERIOD timesteps
        """
        LARGE_NUMBER = 10 ** 6
        components_ask = 0
        components_ask_vol = LARGE_NUMBER
        components_bid = 0
        components_bid_vol = LARGE_NUMBER

        if verbose:
            print("Basket ask price:", basket.getBestAskPrice())
            print("Basket bid price:", basket.getBestBidPrice())

        for i in range(len(components)):
            component = components[i]
            n = amounts[i]
            # if verbose:
            #     print(f"Component {component.PRODUCT} ask price:", component.getBestAskPrice())
            #     print(f"Component {component.PRODUCT} bid price:", component.getBestBidPrice())
            # Compute ask price and volume (price we can buy for)
            components_ask += component.getBestAskPrice() * n
            # How much we would be able to buy based on our current position (and the position limit)
            available_buy_vol = component.POS_LIMIT - component.currentPosition
            # How much we would be able to buy at this price and based on our current position (in number of baskets)
            # eg. if there are 3 components to a basket, and we can buy 6 components, components_ask_vol = 2
            components_ask_vol = min(available_buy_vol / n, component.getBestAskVolume() / n, components_ask_vol)

            # Same as above, but with bid price (price we can sell for)
            components_bid += component.getBestBidPrice() * n
            available_sell_vol = component.POS_LIMIT + component.currentPosition
            components_bid_vol = min(available_sell_vol / n, component.getBestBidVolume() / n, components_bid_vol)

        if verbose:
            print("Components ask price:", components_ask)
            print("Components bid price:", components_bid)

        if is_premium:
            component_mid = (components_ask + components_bid) / 2
            premium = (basket.getMidPrice() - component_mid) / basket.getMidPrice()

            if "basket_premium" in basket.otherStoredVars:
                if len(basket.otherStoredVars["basket_premium"]) < basket.PERIOD:
                    basket.otherStoredVars["basket_premium"] = basket.otherStoredVarsPrev["basket_premium"] + [premium]
                    return
                else:
                    basket.otherStoredVars["basket_premium"] = basket.otherStoredVarsPrev["basket_premium"][1:] + [premium]
            else:
                basket.otherStoredVars["basket_premium"] = [premium]

            predicted_premium = basket.getLR(basket.otherStoredVars["basket_premium"]) * basket.getMidPrice()
            print("predicted_premium:", predicted_premium / basket.getMidPrice())
            print("premium:", premium)

        else:
            predicted_premium = 0


        # You can buy components for less than you can sell the basket for
        if components_ask < basket.getBestBidPrice() - predicted_premium:
            volume = min(components_ask_vol, basket.getBestBidVolume())
            if verbose:
                print("Basket > Components")
                print(f"Selling {volume} {basket.PRODUCT} @ {basket.getBestBidPrice()}")
            # Sell basket
            # basket.orders.append(Order(basket.PRODUCT, basket.getBestBidPrice(), int(-volume)))
            basket.setOrder(basket.getBestBidPrice(), int(volume), "sell")
            # Buy components
            for i in range(len(components)):
                component = components[i]
                n = amounts[i]
                if verbose:
                    print(f"Buying {volume * n} {component.PRODUCT} @ {component.getBestAskPrice()}")
                component.setOrder(component.getBestAskPrice(), int(volume) * n, "buy")
                # component.orders.append(Order(component.PRODUCT, component.getBestAskPrice(), int(volume) * n))

        # You can sell components for more than you can buy the basket for
        if components_bid > basket.getBestAskPrice() - predicted_premium:
            volume = min(components_bid_vol, basket.getBestAskVolume())
            if verbose:
                print("Basket < Components")
                print(f"Buying {volume} {basket.PRODUCT} @ {basket.getBestAskPrice()}")
            # Buy basket
            basket.setOrder(basket.getBestAskPrice(), int(volume), "buy")
            # basket.orders.append(Order(basket.PRODUCT, basket.getBestAskPrice(), int(volume)))
            # Sell components
            for i in range(len(components)):
                component = components[i]
                n = amounts[i]
                if verbose:
                    print(f"Selling {volume * n} {component.PRODUCT} @ {component.getBestBidPrice()}")
                component.setOrder(component.getBestBidPrice(), int(volume) * n, "sell")
                # component.orders.append(Order(component.PRODUCT, component.getBestBidPrice(), int(-volume) * n))

    # endregion

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # region Process traderData (data stored from previous time-steps)
        if state.timestamp > 0:
            prevTraderData = jsonpickle.loads(state.traderData)
            shells_dict = prevTraderData["shells"]
        else:
            shells_dict = {}
            for instrumentInfo in self.allInfo:
                shells_dict[instrumentInfo.PRODUCT] = 0
        # endregion

        # region Set order depth and append live/weighted prices to each instrument
        for instrumentInfo in self.allInfo:
            # Update the variables stored from last iteration
            if state.timestamp > 0:
                instrumentInfo.priceHistory = prevTraderData[instrumentInfo.PRODUCT]["priceHistory"]
                instrumentInfo.otherStoredVarsPrev = prevTraderData[instrumentInfo.PRODUCT]["other"]

            # Update instrumentInfo with currentPosition, orderDepth and HistoricPrice
            try:
                instrumentInfo.currentPosition = state.position[instrumentInfo.PRODUCT]
            except KeyError:
                instrumentInfo.currentPosition = 0
            instrumentInfo.orderDepth = state.order_depths[instrumentInfo.PRODUCT]
            instrumentInfo.appendHistoricPrice()
        # endregion

        # region Process past trades and update shells
        if state.timestamp > 0:
            shells_dict = self.processPastTrades(state, shells_dict)
            self.total_shells = sum(shells_dict.values())
        # endregion

        # region Debug messages
        print("Positions:", state.position)
        if "CROISSANTS" in state.position and "BASKET2" in state.position and "JAMS" in state.position:
            if state.position["CROISSANTS"] != state.position["BASKET2"] * 4 or state.position["JAMS"] != state.position["BASKET2"] * 2:
                print("WARNING")
        print("Shells:", shells_dict)
        print("Total Shells:", self.total_shells)
        # endregion

        # Perform strategies for all products
        for product in state.order_depths.keys():

            if product == "RAINFOREST_RESIN":
                self.fair_price_strat(self.resinInfo, std_devs=0.1, always_transact=1, known_price=10000, verbose=True)

            if product == "KELP":
                self.fair_price_strat(self.kelpInfo, std_devs=0.1, always_transact=1.5, verbose=True)

            if product == "PICNIC_BASKET1":
                self.basket_strategy(self.basket1Info, [self.croissantInfo, self.jamInfo, self.djembeInfo], [6, 3, 1], is_premium=False, verbose=True)

            if product == "PICNIC_BASKET2":
                self.basket_strategy(self.basket2Info, [self.croissantInfo, self.jamInfo], [4, 2], is_premium=True, verbose=True)

        traderData = {}
        result = {}
        for instrumentInfo in self.allInfo:
            # Put in the orders for all products
            result[instrumentInfo.PRODUCT] = instrumentInfo.orders
            instrumentInfo.orders = []

            # Store the variables which need to be stored for future timesteps
            traderData[instrumentInfo.PRODUCT] = {"priceHistory": instrumentInfo.priceHistory, "other": instrumentInfo.otherStoredVars}

        traderData["shells"] = shells_dict

        conversions = 1
        return result, conversions, jsonpickle.dumps(traderData)
