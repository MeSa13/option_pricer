# from numpy import linspace
# import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import copy


class Asset:
    """asset with a given price and volatility"""

    def __init__(self, price, vol):
        self.price = price
        self.vol = vol

    def __str__(self):
        return "current asset price=%s and volatility=%s" % (self.price, self.vol)

    def simulate(self, time, interest_rate, size=None):
        wiener_process = np.random.normal(0.0, 1.0, size)
        if len(wiener_process.shape) == 1:
            number_simulation = 1
            steps = wiener_process.shape[0]
            dt = time / steps
        elif len(wiener_process.shape) == 2:
            number_simulation = wiener_process.shape[0]
            steps = wiener_process.shape[1]
            dt = time / steps
        else:
            return "size is not right!"
        wiener_process.shape = (number_simulation, steps)
        movement_coefficient = 1 + interest_rate * dt + self.vol * np.sqrt(dt) * wiener_process
        simulated_price = np.ones((number_simulation, 1)) * self.price
        for i in range(steps):
            new_price = simulated_price[:, i] * movement_coefficient[:, i]
            new_price.shape = (number_simulation, 1)
            simulated_price = np.append(simulated_price, new_price, axis=1)
        return simulated_price


class Option:
    """option of a given asset, strike price, maturity time and interest rate"""

    def __init__(self, asset, strike_price, maturity_time, interest_rate):
        self.asset = asset
        self.strike_price = strike_price
        self.maturity_time = maturity_time
        self.interest_rate = interest_rate

    def __str__(self):
        return "Option of an asset with price %s, vol %s, strike price %s, maturity time %s year(s) and risk-free " \
               "interest rate %s per year" % (
                   self.asset.price, self.asset.vol, self.strike_price, self.maturity_time, self.interest_rate)


class PutOption(Option):
    """something here!"""

    def __init__(self, asset, strike_price, maturity_time, interest_rate):
        Option.__init__(self, asset, strike_price, maturity_time, interest_rate)

    def analytic_price(self):
        """price of European options using Black-Scholes-Merton analytic formula."""
        d1 = (np.log(self.asset.price / self.strike_price)
              + (self.interest_rate + self.asset.vol ** 2 / 2) * self.maturity_time) \
             / (self.asset.vol * np.sqrt(self.maturity_time))
        d2 = d1 - self.asset.vol * np.sqrt(self.maturity_time)
        price = self.strike_price * np.exp(-self.interest_rate * self.maturity_time) * scipy.stats.norm.cdf(-d2) \
                - self.asset.price * scipy.stats.norm.cdf(-d1)
        return price

    def binomial_price(self, len_chain=10):
        """option pricing using binomial method. len_chain determines
        the length of the binomial tree."""
        if len_chain == 0:  # value at the final node
            return max(self.strike_price - self.asset.price, 0)
        dt = self.maturity_time / len_chain
        a = np.exp(self.interest_rate * dt)  # Q: these numbers do not change.
        # Is there a way to carry them out and not calculate them every time?
        u = np.exp(self.asset.vol * np.sqrt(dt))
        d = np.exp(-self.asset.vol * np.sqrt(dt))
        p = (a - d) / (u - d)
        option_up = copy.deepcopy(self)  # representing the option in up node
        option_down = copy.deepcopy(self)  # representing the option in down node
        option_up.maturity_time -= dt
        option_down.maturity_time -= dt
        option_up.asset.price = u * self.asset.price
        option_down.asset.price = d * self.asset.price
        return (p * option_up.binomial_price(len_chain - 1) + (1 - p) * option_down.binomial_price(len_chain - 1)) / a

    def monte_carlo_price(self, sample=1000, len_chain=100):
        """option pricing using Monte Carlo method.
        sample: number of simulations
        len_chain: length of each simulated chain"""
        simulations = self.asset.simulate(self.maturity_time, self.interest_rate, size=(sample, len_chain))
        asset_price = simulations[:, -1]
        payoff = np.exp(-self.interest_rate * self.maturity_time) * np.maximum(self.strike_price - asset_price, 0)
        price = np.mean(payoff)
        price_std = np.std(payoff)
        return price, price_std


class CallOption(Option):
    """something here!"""

    def __init__(self, asset, strike_price, maturity_time, interest_rate):
        Option.__init__(self, asset, strike_price, maturity_time, interest_rate)

    def analytic_price(self):
        """price of European options using Black-Scholes-Merton analytic formula."""
        d1 = (np.log(self.asset.price / self.strike_price)
              + (self.interest_rate + self.asset.vol ** 2 / 2) * self.maturity_time) / (
                     self.asset.vol * np.sqrt(self.maturity_time))
        d2 = d1 - self.asset.vol * np.sqrt(self.maturity_time)
        price = - self.strike_price * np.exp(-self.interest_rate * self.maturity_time) * scipy.stats.norm.cdf(d2) \
                + self.asset.price * scipy.stats.norm.cdf(d1)
        return price

    def binomial_price(self, len_chain=10):
        """option pricing using binomial method. len_chain determines
        the length of the binomial tree."""
        if len_chain == 0:  # value at the final node
            return max(self.asset.price - self.strike_price, 0)
        dt = self.maturity_time / len_chain
        a = np.exp(self.interest_rate * dt)  # Q: these numbers do not change.
        # Is there a way to carry them out and not calculate them every time?
        u = np.exp(self.asset.vol * np.sqrt(dt))
        d = np.exp(-self.asset.vol * np.sqrt(dt))
        p = (a - d) / (u - d)
        option_up = copy.deepcopy(self)  # representing the option in up node
        option_down = copy.deepcopy(self)  # representing the option in down node
        # does not work with copy.copy!
        option_up.maturity_time -= dt
        option_down.maturity_time -= dt
        option_up.asset.price = u * self.asset.price
        option_down.asset.price = d * self.asset.price
        return (p * option_up.binomial_price(len_chain - 1) + (1 - p) * option_down.binomial_price(len_chain - 1)) / a

    def monte_carlo_price(self, sample=1000, len_chain=100):
        """option pricing using Monte Carlo method.
        sample: number of simulations
        len_chain: length of each simulated chain"""
        simulations = self.asset.simulate(self.maturity_time, self.interest_rate, size=(sample, len_chain))
        asset_price = simulations[:, -1]
        payoff = np.exp(-self.interest_rate * self.maturity_time) * np.maximum(asset_price - self.strike_price, 0)
        price = np.mean(payoff)
        price_std = np.std(payoff)
        return price, price_std


stock = Asset(100.0, 0.3)
call = CallOption(stock, 100.0, 0.2, 0.1)
put = PutOption(stock, 100.0, 0.2, 0.1)
print(call.analytic_price(), put.analytic_price())
print(call.binomial_price(), put.binomial_price())
print(call.monte_carlo_price(), put.monte_carlo_price())
print("Done!")
