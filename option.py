"""option.py
~~~~~~~~~~~~~~~~~~~~~
Pricing financial derivatives, particularly options.
For each type of option (European, American, etc.), a range of pricing methods
are introduced and Greeks are calculated.

"""


### Libraries

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import scipy.optimize
from abc import ABC, abstractmethod


class Asset:
    """Make Asset class

    Attributes:
        price (float): asset price
        vol (function): local volatility of the asset
    """

    def __init__(self, price, vol):
        """Initialize Asset"""

        self.price = price
        # if a float is passed for vol (constant volatility) make it a callable object
        self.vol = vol if callable(vol) else lambda x, y: vol

    def simulate(self, duration, growth_rate, number_simulations, steps):
        """Simulate a geometric Brownian motion of the asset price

        Args:
            duration (float): time duration of the simulation
            growth_rate (float): growth rate (per unit time) of the geometric Brownian motion
            number_simulations (int): number of simulations
            steps (int): number of steps in each simulation
        Returns:
            sim_price (numpy.ndarray): an array of size number_simulations * (step+1)
        """
        normal_seeds = np.random.normal(0.0, 1.0, (number_simulations, steps))
        dt = duration / steps
        # Current asset price is the first column in sim_price
        sim_price = np.ones((number_simulations, 1)) * self.price
        t = 0
        for i in range(steps):
            growth_coeff = 1.0 + growth_rate * dt + self.vol(sim_price[:, i], t) * np.sqrt(dt) * normal_seeds[:, i]
            new_price = sim_price[:, i] * growth_coeff
            # making new_price a column vector
            new_price.shape = (number_simulations, 1)
            sim_price = np.append(sim_price, new_price, axis=1)
            t += dt
        return sim_price

    def sim_plot(self, growth_rate, duration, step=100, num_sim=5):
        """Plot simulation of the asset price movement

        Args:
            growth_rate (float): growth rate of the asset price, usually taken to be the free interest rate
            duration (float): duration of simulation
            step (int): number of time steps in each simulation, defaults to 100
            num_sim (int): number of simulations, defaults to 5
        """
        t = np.arange(0.0, duration, duration/step)
        sim_price = self.simulate(duration, growth_rate, num_sim, len(t) - 1)
        plt.plot(t, np.transpose(sim_price))
        plt.xlabel("time")
        plt.ylabel("asset price")
        plt.show()

    def asset_tree(self, t, step):
        """Form asset price tree for binomial pricing

        Args:
            t (float): time duration
            step (integer): number of steps in the tree
        Returns:
        (step+1)*(step+1) matrix of prices
        """
        dt = 1.0 * t / step
        # making the binomial tree as a matrix. [i,j] element represents the price after i up movements and j
        # down movements in price of the asset. Elements with i+j=n correspond to the nth layer of the tree.

        # making the asset price tree.
        tree = np.zeros((step + 1, step + 1))
        tree[0, 0] = self.price
        # evaluating elements resulting from only up movements in the asset price ([i,0]).
        for up_mov in range(1, step + 1):
            # up movement factor in price of the asset
            u = np.exp(self.vol(tree[up_mov - 1, 0], (up_mov - 1) * dt) * np.sqrt(dt))
            tree[up_mov, 0] = tree[up_mov - 1, 0] * u
        # evaluating elements resulting from only down movements in the asset price ([0,i]).
        for down_mov in range(1, step + 1):
            # down movement factor in price of the asset
            d = np.exp(-self.vol(tree[0, down_mov - 1], (down_mov - 1) * dt) * np.sqrt(dt))
            tree[0, down_mov] = tree[0, down_mov - 1] * d
        # evaluating the whole asset_tree, going by each layer of the tree
        for layer in range(1, step + 1):
            for up_mov in range(1, layer):
                down_mov = layer - up_mov
                u = np.exp(self.vol(tree[up_mov - 1, down_mov], (layer - 1) * dt) * np.sqrt(dt))
                d = np.exp(-self.vol(tree[up_mov, down_mov - 1], (layer - 1) * dt) * np.sqrt(dt))
                # taking the average of the price coming from one up movement or one down movement before. For
                # constant volatility these two are the same. For local volatility depending on asset price
                # they are different.
                tree[up_mov, down_mov] = (u * tree[up_mov - 1, down_mov] + d * tree[up_mov, down_mov - 1]) / 2.0
        return tree


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

    def payoff(self, s):
        """Return Payoff"""
        if self.option_type == "Put":
            return np.maximum(self.strike - s, 0)
        elif self.option_type == "Call":
            return np.maximum(s - self.strike, 0)

    @abstractmethod
    def price(self, *args):
        pass

    def delta(self, *args, eps=0.01):
        """Calculate delta of the option

        Args:
            *args:
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
        return (price2 + price3 - 2.0 * price1) / eps ** 2

    def vega(self, *args, eps=0.01):
        """Calculate vega of the option. This method assumes constant volatility.

        Args:
            *args:
            eps (float): finite difference derivative. Defaults to 0.01
        Returns:
            vega of the option
        """
        # Fix the random generator seed for MonteCarlo pricing
        st = np.random.get_state()
        price1 = self.price(*args)
        vol_function = self.asset.vol
        self.asset.vol = lambda x,y: eps + vol_function(self.asset.price, 0.0)
        np.random.set_state(st)
        price2 = self.price(*args)
        self.asset.vol = vol_function
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
        return (price2 - price1) / (365 * eps)


class EuropeanOption(Option):
    """Form a European Option of a given Asset.

    Attributes:
        option_type (str): "Call" or "Put"
        asset (Asset): underlying asset of the option
        strike (float): option strike price
        term (float): option maturity time
        interest_rate (float): risk free interest rate
        pricing method (str): the main pricing method for the option, "Binomial", "MonteCarlo" or "Analytic"
    """

    def analytic(self):
        """Price a European option with Black-Scholes-Merton analytic formula, assuming a constant volatility."""
        vol = self.asset.vol(1.0, 0.0)
        d1 = (np.log(self.asset.price / self.strike)
              + (self.interest_rate + vol ** 2 / 2) * self.term) \
             / (vol * np.sqrt(self.term))
        d2 = d1 - vol * np.sqrt(self.term)
        if self.option_type == "Put":
            return self.strike * np.exp(-self.interest_rate * self.term) * scipy.stats.norm.cdf(-d2) \
                   - self.asset.price * scipy.stats.norm.cdf(-d1)
        elif self.option_type == "Call":
            return - self.strike * np.exp(-self.interest_rate * self.term) * scipy.stats.norm.cdf(d2) \
                   + self.asset.price * scipy.stats.norm.cdf(d1)

    def binomial(self, step=1000):
        """Price a European option with Binomial tree method

        Args:
            step (int): number of layers in the tree. Defaults to 1000.
        Returns:
            price of the European option (float)
        """
        dt = 1.0 * self.term / step
        # making the binomial tree as a matrix. [i,j] element represents the price after i up movements and j
        # down movements in price of the asset. Elements with i+j=n correspond to the nth layer of the tree.

        # making the asset price tree.
        asset_tree = self.asset.asset_tree(self.term, step)
        a = np.exp(self.interest_rate * dt)
        # making the option price tree.
        option_tree = np.zeros((step + 1, step + 1))
        # pricing the option at maturity. [i,j] elements with i+j=step correspond to maturity.
        for up_mov in range(step + 1):
            option_tree[up_mov, step - up_mov] = self.payoff(asset_tree[up_mov, step - up_mov])
        # pricing the option from the maturity layer backwards
        for layer in range(step - 1, -1, -1):
            for up_mov in range(layer + 1):
                down_mov = layer - up_mov
                u = np.exp(self.asset.vol(asset_tree[up_mov, down_mov], layer * dt) * np.sqrt(dt))
                d = np.exp(-self.asset.vol(asset_tree[up_mov, down_mov], layer * dt) * np.sqrt(dt))
                # probability of up movement in the price of the asset
                p = (a - d) / (u - d)
                option_tree[up_mov, down_mov] = (p * option_tree[up_mov + 1, down_mov]
                                                  + (1 - p) * option_tree[up_mov, down_mov + 1]) / a
        return option_tree[0, 0]

    def binomial_constant_vol(self, step=1000):
        """
        Price a European option with Binomial tree method when volatility is constant.
        For a constant volatility, this method is faster compared to ``binomial`` method. 

        Args:
            step (int): number of layers in the tree. Defaults to 1000.
        Returns:
            price of the European option (float)
        """
        vol = self.asset.vol(1.0, 0.0)
        dt = 1.0 * self.term / step
        a = np.exp(self.interest_rate * dt)
        # up movement factor in price of the asset
        u = np.exp(vol * np.sqrt(dt))
        # down movement factor in price of the asset
        d = np.exp(-vol * np.sqrt(dt))
        # probability of up movement in price of the asset
        p = (a - d) / (u - d)
        # making the binomial tree as a matrix. [i,j] element represents the option price after i up movements and j
        # down movements in price of the asset. Elements with i+j=n correspond to the nth layer of the tree.
        option_price = np.zeros((step + 1, step + 1))
        # pricing the option at maturity. [i,j] elements with i+j=step correspond to maturity.
        for up_mov in range(step + 1):
            option_price[up_mov, step - up_mov] = self.payoff(u ** up_mov * d ** (step - up_mov) * self.asset.price)
        # pricing the option from maturity layer backwards
        for layer in range(step - 1, -1, -1):
            for up_mov in range(layer + 1):
                down_mov = layer - up_mov
                option_price[up_mov, down_mov] = (p * option_price[up_mov + 1, down_mov]
                                                  + (1 - p) * option_price[up_mov, down_mov + 1]) / a
        return option_price[0, 0]

    def monte_carlo(self, number_simulation=10000, steps=100):
        """Price a European option using Monte Carlo method.

        Args:
            number_simulation (int): number of Monte Carlo simulations. Defaults to 10,000.
            steps (int): length of each Monte Carlo simulations. Defaults to 100.
        Returns:
            price of the European option (float)
        """
        simulations = self.asset.simulate(self.term, self.interest_rate, number_simulation, steps)
        # list of asset prices at maturity
        asset_price_list = simulations[:, -1]
        payoff_list = self.payoff(asset_price_list)
        price = np.exp(-self.interest_rate * self.term) * np.mean(payoff_list)
        return price

    def pde(self, dx, grid_size=4.0, ratio=1.0):
        """Price a European option by solving Black-Scholes differential equation

        Args:
            dx (float): spatial grid separation
            grid_size (float): size of spatial grid, defaults to 4
            ratio (float): controlling time step size
        Returns:
            price of the European option (float)
        """
        # a change of variable in Black-Scholes equation is performed, as x= ln(s), s = asset price.
        dt = ratio * dx ** 2
        t = self.term
        # Normalizing prices by strike, to work with order 1 numbers.
        normalized_price = self.asset.price / self.strike
        # normalized strike is 1.
        k = 1.0
        # [-n, n] range in x corresponds to [k e^(-n), k e^n] in s.
        x = np.arange(-grid_size, grid_size + dx, dx)
        # evaluating option price at maturity
        if self.option_type == "Call":
            option_price = np.maximum(np.exp(x) - k, 0)
        else:
            option_price = np.maximum(k - np.exp(x), 0)
        while t > 0:
            temp_price = option_price
            t -= dt
            # updating option price. The first and last element in option_price
            # need to be updated by boundary conditions.
            for i in range(1, len(x) - 1):
                sigma = self.asset.vol(np.exp(x[i]), t)
                option_price[i] = temp_price[i] * (1 - sigma ** 2 * ratio - self.interest_rate * dt) \
                                  + 0.5 * temp_price[i + 1] * \
                                  (sigma ** 2 * ratio + (self.interest_rate - 0.5 * sigma ** 2) * ratio * dx) \
                                  + 0.5 * temp_price[i - 1] * \
                                  (sigma ** 2 * ratio - (self.interest_rate - 0.5 * sigma ** 2) * ratio * dx)
            # updating boundary values
            if self.option_type == "Put":
                option_price[0] *= np.exp(-self.interest_rate * dt)
        # corresponding value to the asset price in terms of x
        x0 = np.log(normalized_price)
        # taking the average of two elements closest to x0 in option_price and renormalize the price
        price = self.strike * (option_price[x > x0][0] + option_price[x < x0][-1]) / 2.0
        return price

    def price(self, *args):
        """Return option price using the method specified by pricing_method"""
        if self.pricing_method == "Binomial":
            return self.binomial(*args)
        elif self.pricing_method == "MonteCarlo":
            return self.monte_carlo(*args)
        elif self.pricing_method == "Analytic":
            return self.analytic()

    def implied_vol(self, realized_price, top_vol=1.0, bot_vol=0.001):
        """Calculate the implied volatility of an option using bisection method

        This method solves for the volatility of an option given its price.
        Args:
            realized_price (float): traded price of the option
            top_vol (float): high value of volatility in bisection method
            bot_vol (float): low value of volatility in bisection method
        Returns:
            implied volatility of the option (float)
        """
        # Make the target function to look for its root
        def target_function(sigma):
            asset = Asset(self.asset.price, sigma)
            eu_option = EuropeanOption(self.option_type, asset, self.strike, self.term, self.interest_rate, "Analytic")
            return eu_option.price() - realized_price
        return scipy.optimize.bisect(target_function, top_vol, bot_vol)


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
        Option.__init__(self, option_type, asset, strike, term, interest_rate, "Binomial")

    def binomial(self, step=1000):
        """Price an American option with Binomial tree method

        Args:
            step (int): number of layers in the tree. Defaults to 1000.
        Returns:
            price of the American option
        """
        dt = 1.0 * self.term / step
        # making the binomial tree as a matrix. [i,j] element represents the price after i up movements and j
        # down movements in price of the asset. Elements with i+j=n correspond to the nth layer of the tree.

        # making the asset price tree.
        asset_tree = self.asset.asset_tree(self.term, step)
        a = np.exp(self.interest_rate * dt)
        # making the option price tree.
        option_price = np.zeros((step + 1, step + 1))
        # pricing the option at maturity. [i,j] elements with i+j=step correspond to maturity.
        for up_mov in range(step + 1):
            option_price[up_mov, step - up_mov] = self.payoff(asset_tree[up_mov, step - up_mov])
        # pricing the option from the maturity layer backwards
        for layer in range(step - 1, -1, -1):
            for up_mov in range(layer + 1):
                down_mov = layer - up_mov
                u = np.exp(self.asset.vol(asset_tree[up_mov, down_mov], layer * dt) * np.sqrt(dt))
                d = np.exp(-self.asset.vol(asset_tree[up_mov, down_mov], layer * dt) * np.sqrt(dt))
                # probability of up movement in the price of the asset
                p = (a - d) / (u - d)
                option_price[up_mov, down_mov] = (p * option_price[up_mov + 1, down_mov]
                                                  + (1 - p) * option_price[up_mov, down_mov + 1]) / a
                current_payoff = self.payoff(asset_tree[up_mov, down_mov])
                option_price[up_mov, down_mov] = max(current_payoff, option_price[up_mov, down_mov])
        return option_price[0, 0]

    def price(self, *args):
        """Price American option"""
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
        average_period_index = 1 + int((1 - self.average_period / self.term) * steps)
        # asset price list during average_period
        average_period_asset_price_list = simulations[:, average_period_index:]
        # list of average asset price during average_period
        average_period_average_asset_price_list = np.mean(average_period_asset_price_list, axis=1)
        payoff_list = self.payoff(average_period_average_asset_price_list)
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

    def price(self, *args):
        """Price Asian option"""
        return self.monte_carlo(*args)

    def effective_vol(self, top_vol=1.0, bot_vol=0.001, *args):
        """Calculate the volatility of a European option with similar properties and price of the Asian option"""
        price = self.monte_carlo_error(*args)[0]
        asset = Asset(self.asset.price, 1.0)
        eu_option = EuropeanOption(self.option_type, asset, self.strike, self.term, self.interest_rate, "Analytic")
        return eu_option.implied_vol(price, top_vol=top_vol, bot_vol=bot_vol)

