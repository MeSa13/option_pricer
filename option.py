# from numpy import linspace
# import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import time
from abc import ABC, abstractmethod


class Asset:
    """Make Asset class

    Attributes:
        price (float): asset price
        vol (float): volatility of the asset. For example, 0.1 means 10% volatility
    """

    def __init__(self, price, vol):
        """Initialize Asset

        Args:
            price (float): asset price
            vol (float): asset volatility
        """
        self.price = price
        self.vol = vol

    def simulate(self, duration, growth_rate, number_simulations, steps):
        """Simulate a geometric Brownian motion of the asset price

        Args:
            duration (float): time duration of the simulation
            growth_rate (float): growth rate of the geometric Brownian motion
            number_simulations (int): number of simulations
            steps (int): number of steps in each simulation
        Returns:
            simulated_price (numpy.ndarray): an array of size number_simulations * (step+1)
        """
        random_normal_seeds = np.random.normal(0.0, 1.0, (number_simulations, steps))
        dt = duration / steps
        movement_coefficient = 1.0 + growth_rate * dt + self.vol * np.sqrt(dt) * random_normal_seeds
        # Current asset price is the first column in simulated_price
        simulated_price = np.ones((number_simulations, 1)) * self.price
        for i in range(steps):
            new_price = simulated_price[:, i] * movement_coefficient[:, i]
            new_price.shape = (number_simulations, 1)
            simulated_price = np.append(simulated_price, new_price, axis=1)
        return simulated_price


class Option(ABC):
    """Form an Option of a given Asset

    Attributes:
        option_type (str): "Call" or "Put"
        asset (Asset): underlying asset of the option
        strike (float): option strike price
        term (float): option maturity time
        interest_rate (float): risk free interest rate
        pricing method (str): the main pricing method for the option, "Binomial", "MonteCarlo" or "Analytic"
    """

    def __init__(self, option_type, asset, strike, term, interest_rate, pricing_method):
        """Initialize Option

        Args:
            option_type (str): "Call" or "Put"
            asset (Asset): underlying asset of the option
            strike (float): option strike price
            term (float): option maturity time
            interest_rate (float): risk free interest rate
            pricing method (str): the main pricing method for the option, "Binomial", "MonteCarlo" or "Analytic"
        """
        self.option_type = option_type
        assert self.option_type in ["Call", "Put"]
        self.asset = asset
        self.strike = strike
        self.term = term
        self.interest_rate = interest_rate
        self.pricing_method = pricing_method
        assert self.pricing_method in ["Binomial", "MonteCarlo", "Analytic"]

    def payoff(self):
        """Return Payoff function"""
        if self.option_type is "Put":
            return lambda s: np.maximum(self.strike - s, 0)
        elif self.option_type is "Call":
            return lambda s: np.maximum(s - self.strike, 0)

    @abstractmethod
    def price(self, *args):
        pass

    def delta(self, *args, eps=0.01):
        """Calculate delta of the option

        Args:
            *args: QQ
            eps (float): finite difference derivative. Defaults to 0.01
        Returns:
            delta of the option
        """
        # Fix the random generator seed for MonteCarlo pricing
        st = np.random.get_state()
        price1 = self.price(*args)
        self.asset.price += eps
        np.random.set_state(st)
        price2 = self.price(*args)
        self.asset.price -= eps
        return (price2 - price1) / eps

    def gamma(self, *args, eps=2.0):
        """Calculate gamma of the option

        Args:
            *args:
            eps (float): finite difference derivative. Defaults to 2.0
        Returns:
            gamma of the option
        """
        # Fix the random generator seed for MonteCarlo pricing
        st = np.random.get_state()
        price1 = self.price(*args)
        np.random.set_state(st)
        self.asset.price += eps
        price2 = self.price(*args)
        np.random.set_state(st)
        self.asset.price -= 2 * eps
        price3 = self.price(*args)
        self.asset.price += eps
        return (price2 + price3 - 2.0*price1) / eps ** 2

    def vega(self, *args, eps=0.01):
        """Calculate vega of the option

        Args:
            *args:
            eps (float): finite difference derivative. Defaults to 0.01
        Returns:
            vega of the option
        """
        # Fix the random generator seed for MonteCarlo pricing
        st = np.random.get_state()
        price1 = self.price(*args)
        self.asset.vol += eps
        np.random.set_state(st)
        price2 = self.price(*args)
        self.asset.vol -= eps
        return 0.01 * (price2 - price1) / eps

    def rho(self, *args, eps=0.01):
        """Calculate rho of the option

        Args:
            *args:
            eps (float): finite difference derivative. Defaults to 0.01
        Returns:
            rho of the option per 1% change in interest rate
        """
        # Fix the random generator seed for MonteCarlo pricing
        st = np.random.get_state()
        price1 = self.price(*args)
        self.interest_rate += eps
        np.random.set_state(st)
        price2 = self.price(*args)
        self.interest_rate -= eps
        return 0.01 * (price2 - price1) / eps

    def theta(self, *args, eps=0.001):
        """Calculate theta of the option

        Args:
            *args:
            eps (float): finite difference derivative. Defaults to 0.001
        Returns:
            theta of the option per calendar day
        """
        # Fix the random generator seed for MonteCarlo pricing
        st = np.random.get_state()
        price1 = self.price(*args)
        np.random.set_state(st)
        self.term -= eps
        price2 = self.price(*args)
        self.term += eps
        return (price1 - price2) / (365*eps)


class EuropeanOption(Option):
    """Form a European Option of a given Asset.

    QQ: how to refer inheritance? Should I repeat attributes?

    Attributes:
        option_type (str): "Call" or "Put"
        asset (Asset): underlying asset of the option
        strike (float): option strike price
        term (float): option maturity time
        interest_rate (float): risk free interest rate
        pricing method (str): the main pricing method for the option, "Binomial", "MonteCarlo" or "Analytic"
    """

    def analytic(self):
        """Price a European option with Black-Scholes-Merton analytic formula."""

        d1 = (np.log(self.asset.price / self.strike)
              + (self.interest_rate + self.asset.vol ** 2 / 2) * self.term) \
             / (self.asset.vol * np.sqrt(self.term))
        d2 = d1 - self.asset.vol * np.sqrt(self.term)
        if self.option_type is "Put":
            option_price = self.strike * np.exp(-self.interest_rate * self.term) * scipy.stats.norm.cdf(-d2) \
                    - self.asset.price * scipy.stats.norm.cdf(-d1)
        elif self.option_type is "Call":
            option_price = - self.strike * np.exp(-self.interest_rate * self.term) * scipy.stats.norm.cdf(d2) \
                    + self.asset.price * scipy.stats.norm.cdf(d1)
        return option_price

    def binomial(self, step=1000):
        """Price a European option with Binomial tree method

        Args:
            step (int): number of layers in the tree. Defaults to 1000.
        Returns:
            price of the European option
        """
        dt = 1.0 * self.term / step
        a = np.exp(self.interest_rate * dt)
        # up movement factor in price of the asset
        u = np.exp(self.asset.vol * np.sqrt(dt))
        # down movement factor in price of the asset
        d = np.exp(-self.asset.vol * np.sqrt(dt))
        # probability of up movement in price of the asset
        p = (a - d) / (u - d)
        # making the binomial tree as a matrix. [i,j] element represents the option price after i up movements and j
        # down movements in price of the asset. Elements with i+j=n correspond to the nth layer of the tree.
        option_price = np.zeros((step+1, step+1))
        # pricing the option at maturity. [i,j] elements with i+j=step correspond to maturity.
        for up_mov in range(step + 1):
            option_price[up_mov, step - up_mov] = self.payoff()(u ** up_mov * d ** (step - up_mov) * self.asset.price)
        # pricing the option from maturity layer backwards
        for layer in range(step - 1, -1, -1):
            for up_mov in range(layer + 1):
                down_mov = layer - up_mov
                option_price[up_mov, down_mov] = (p * option_price[up_mov + 1, down_mov] \
                                                  + (1 - p) * option_price[up_mov, down_mov + 1]) / a
        return option_price[0, 0]

    def monte_carlo(self, number_simulation=10000, steps=100):
        """Price a European option using Monte Carlo method.

        Args:
            number_simulation (int): number of Monte Carlo simulations. Defaults to 10,000.
            steps (int): length of each Monte Carlo simulations. Defaults to 100.
        Returns:
            price of the European option
        """
        simulations = self.asset.simulate(self.term, self.interest_rate, number_simulation, steps)
        # list of asset prices at maturity QQ: should I include comments like this
        asset_price_list = simulations[:, -1]
        payoff_list = self.payoff()(asset_price_list)
        price = np.exp(-self.interest_rate * self.term) * np.mean(payoff_list)
        return price

    def price(self, *args):
        """Return option price using the method specified by pricing_method""" #QQ
        if self.pricing_method == "Binomial":
            return self.binomial(*args)
        elif self.pricing_method == "MonteCarlo":
            return self.monte_carlo(*args)
        elif self.pricing_method == "Analytic":
            return self.analytic()


class AmericanOption(Option):
    """Form an AmericanOption of a given Asset.

    Attributes:
        option_type (str): "Call" or "Put"
        asset (Asset): underlying asset of the option
        strike (float): option strike price
        term (float): option maturity time
        interest_rate (float): risk free interest rate
    """
    def __init__(self, option_type, asset, strike, term, interest_rate):
        """Initialize AmericanOption"""
        #QQ: anything else?!
        Option.__init__(self, option_type, asset, strike, term, interest_rate, "Binomial")

    def binomial(self, step=1000):
        """Price an American option with Binomial tree method

        Args:
            step (int): number of layers in the tree. Defaults to 1000.
        Returns:
            price of the European option
        """
        dt = 1.0 * self.term / step
        a = np.exp(self.interest_rate * dt)
        # up movement factor in price of the asset
        u = np.exp(self.asset.vol * np.sqrt(dt))
        # down movement factor in price of the asset
        d = np.exp(-self.asset.vol * np.sqrt(dt))
        # probability of up movement in price of the asset
        p = (a - d) / (u - d)
        # making the binomial tree as a matrix. [i,j] element represents the option price after i up movements and j
        # down movements in price of the asset. Elements with i+j=n correspond to the nth layer of the tree.
        option_price = np.zeros((step+1, step+1))
        # pricing the option at maturity. [i,j] elements with i+j=step correspond to maturity.
        for up_mov in range(step + 1):
            option_price[up_mov, step - up_mov] = self.payoff()(u ** up_mov * d ** (step - up_mov) * self.asset.price)
        # pricing the option from the maturity layer backwards
        for layer in range(step - 1, -1, -1):
            for up_mov in range(layer + 1):
                down_mov = layer - up_mov
                option_price[up_mov, down_mov] = (p * option_price[up_mov + 1, down_mov] \
                                                  + (1 - p) * option_price[up_mov, down_mov + 1]) / a
                current_payoff = self.payoff()(u ** up_mov * d ** down_mov * self.asset.price)
                option_price[up_mov, down_mov] = max(current_payoff, option_price[up_mov, down_mov])
        return option_price[0, 0]

    def price(self, *args):
        return self.binomial(*args)


class AsianOption(Option):
    """Form an AsianOption of a given Asset.

    Attributes:
        option_type (str): "Call" or "Put"
        asset (Asset): underlying asset of the option
        strike (float): option strike price
        term (float): option maturity time
        interest_rate (float): risk free interest rate
        average_period (float): duration of the average period before maturity
    """

    def __init__(self, option_type, asset, strike, term, interest_rate, average_period):
        assert average_period <= term, "average_period should be less than or equal term"
        self.average_period = average_period
        Option.__init__(self, option_type, asset, strike, term, interest_rate, "MonteCarlo")

    def monte_carlo(self, number_simulation=10000, steps=100):
        """Price an Asian option using Monte Carlo method.

        Args:
            number_simulation (int): number of Monte Carlo simulations. Defaults to 10,000.
            steps (int): length of each Monte Carlo simulations. Defaults to 100.
        Returns:
            price of the Asian option
        """
        simulations = self.asset.simulate(self.term, self.interest_rate, number_simulation, steps)
        # index in simulations corresponding to the beginning of average_period time
        average_period_index = 1 + int((1 - self.average_period/self.term) * steps)
        # asset price list during average_period
        average_period_asset_price_list = simulations[:, average_period_index:]
        # list of average asset price during average_period
        average_period_average_asset_price_list = np.mean(average_period_asset_price_list, axis=1)
        payoff_list = self.payoff()(average_period_average_asset_price_list)
        price = np.exp(-self.interest_rate * self.term) * np.mean(payoff_list)
        return price

    def monte_carlo_error(self, number_simulation=10000, steps=100, sample=10):
        """Calculate the error of MonteCarlo pricing method
        Args:
            number_simulation (int): number of Monte Carlo simulations. Defaults to 10,000.
            steps (int): length of each Monte Carlo simulations. Defaults to 100.
            sample (int): number of MonteCarlo runs. Defaults to 10.
        Returns:
            price of the Asian option and its Monte Carlo error.
        """
        price_list = []
        for i in range(sample):
            price_list.append(self.monte_carlo(number_simulation, steps))
        return np.mean(price_list), np.std(price_list)

    def price(self, *args, error=False):
        if error:
            return self.monte_carlo_error(*args)
        else:
            return self.monte_carlo(*args)


ex_asset_price = 100.0
ex_strike = 100.0
ex_vol = 0.3
ex_term = 1.0
ex_interest = 0.1
ex_stock = Asset(ex_asset_price, ex_vol)
eu_call_analytic = EuropeanOption("Call", ex_stock, ex_strike, ex_term, ex_interest, "Analytic")
eu_call_binomial = EuropeanOption("Call", ex_stock, ex_strike, ex_term, ex_interest, "Binomial")
eu_call_MC = EuropeanOption("Call", ex_stock, ex_strike, ex_term, ex_interest, "MonteCarlo")
eu_put_analytic = EuropeanOption("Put", ex_stock, ex_strike, ex_term, ex_interest, "Analytic")
eu_put_binomial = EuropeanOption("Put", ex_stock, ex_strike, ex_term, ex_interest, "Binomial")
eu_put_MC = EuropeanOption("Put", ex_stock, ex_strike, ex_term, ex_interest, "MonteCarlo")
american_call = AmericanOption("Call", ex_stock, ex_strike, ex_term, ex_interest)
american_put = AmericanOption("Put", ex_stock, ex_strike, ex_term, ex_interest)

ex_average_period = 0.1
asian_call = AsianOption("Call", ex_stock, ex_strike, ex_term, ex_interest, ex_average_period)
asian_put = AsianOption("Put", ex_stock, ex_strike, ex_term, ex_interest, ex_average_period)

print("option price")
print("European")
print(eu_call_analytic.price(), eu_put_analytic.price())
# print(eu_call_binomial.price(), eu_put_binomial.price())
# print(eu_call_MC.price(), eu_put_MC.price())
# print("American")
# print(american_call.price(), american_put.price(), "\n")
print("Asian")
print(asian_call.price(), asian_put.price())
#
# print("delta")
# print(eu_call_analytic.delta(), eu_put_analytic.delta())
# print(eu_call_binomial.delta(), eu_put_binomial.delta())
# print(eu_call_MC.delta(), eu_put_MC.delta(), "\n")
#
# print("gamma")
# print(eu_call_analytic.gamma(), eu_put_analytic.gamma())
# print(eu_call_binomial.gamma(), eu_put_binomial.gamma())
# print(eu_call_MC.gamma(), eu_put_MC.gamma(), "\n")
#
# print("vega")
# print(eu_call_analytic.vega(), eu_put_analytic.vega())
# print(eu_call_binomial.vega(), eu_put_binomial.vega())
# print(eu_call_MC.vega(), eu_put_MC.vega(), "\n")
#
# print("rho")
# print(eu_call_analytic.rho(), eu_put_analytic.rho())
# print(eu_call_binomial.rho(), eu_put_binomial.rho())
# print(eu_call_MC.rho(), eu_put_MC.rho(), "\n")
#
# print("theta")
# print(eu_call_analytic.theta(), eu_put_analytic.theta())
# print(eu_call_binomial.theta(), eu_put_binomial.theta())
# print(eu_call_MC.theta(), eu_put_MC.theta(), "\n")
