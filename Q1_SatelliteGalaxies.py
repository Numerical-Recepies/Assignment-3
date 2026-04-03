# imports
from math import gamma
import os
from typing import Callable, Any

import numpy as np
import matplotlib.pyplot as plt

X_MAX = 5.0


# =====================================================
# ========== Helper functions and classes =============
# =====================================================

class MCMC:
    """
    A class to perform Markov Chain Monte Carlo (MCMC) sampling
    using the Metropolis-Hastings algorithm.
    """

    def __init__(
        self,
        likelihood: Callable[..., Any],
        proposed_distribution: Callable[..., Any],
        **kwargs_proposal_dist,
    ):
        """
        Initializes the MCMC class.

        :param likelihood: A function that evaluates the likelihood
        for a given set of parameters and data.
        :type likelihood: Callable[..., Any]
        :param proposed_distribution: A function that generates trial
        samples for the proposal distribution.
        :type proposed_distribution: Callable[..., Any]
        :param kwargs_proposal_dist: Additional arguments to pass to
        the proposal distribution function.
        :type kwargs_proposal_dist: dict
        """
        self._likelihood = likelihood
        self._proposed_distribution = proposed_distribution
        self._proposed_distribution_kwargs = kwargs_proposal_dist

    def metropolis_hastings(
        self,
        initial_sample: np.ndarray,
        data: tuple[np.ndarray, np.ndarray],
        num: int = 100000,
        burn_in: int = 1000,
    ):
        """
        Executes the Metropolis-Hastings algorithm for MCMC sampling.

        :param initial_sample: The initial parameter values for the chain.
        :type initial_sample: np.ndarray
        :param data: The observed data used in the likelihood function.
        :type data: tuple[np.ndarray, np.ndarray]
        :param num: The total number of samples to generate
        (default is 100000).
        :type num: int, optional
        :param burn_in: The number of initial samples to discard as burn-in
        (default is 1000).
        :type burn_in: int, optional
        :return: The retained samples after the burn-in period.
        :rtype: list[np.ndarray]
        """
        samples = [initial_sample]

        num_accepted_samples = 0
        num_total_samples = 0

        for _ in range(num):
            trial_sample = self._proposed_distribution(
                samples[-1], **self._proposed_distribution_kwargs
            )
            trial_sample = np.array(trial_sample[0])

            a = self._likelihood(
                parameters=trial_sample, data=data
            ) / self._likelihood(parameters=samples[-1], data=data)

            if a > 1:
                samples.append(trial_sample)
                num_accepted_samples += 1
            else:
                rand_num = np.random.random()
                if rand_num < a:
                    samples.append(trial_sample)
                    num_accepted_samples += 1
                else:
                    samples.append(samples[-1])

            num_total_samples += 1

        self._print_analytics(num_accepted_samples, num_total_samples)

        retained_samples = samples[burn_in:]

        return retained_samples

    def _print_analytics(
        self, num_accepted_samples: int, num_total_samples: int
    ):
        """
        Prints summary statistics of the sampling process.

        :param num_accepted_samples: The number of samples
        accepted during sampling.
        :type num_accepted_samples: int
        :param num_total_samples: The total number of
        samples attempted.
        :type num_total_samples: int
        """
        acceptance_rate = (num_accepted_samples / num_total_samples) * 100
        header = "MCMC - Metropolis Hastings analytics"
        line_length = 49

        def format_line(content: str) -> str:
            """
            Formats a string to be left-justified within a
            fixed-width space and wraps it with vertical bars.

            :param content: The text to format.
            :type content: str
            :return: The formatted string, padded with spaces
            and wrapped in vertical bars.
            :rtype: str
            """
            return f"| {content.ljust(line_length - 4)} |"

        print("-" * line_length)
        print(format_line(header.center(line_length - 4)))
        print("-" * line_length)
        print(
            format_line(f"Number of accepted samples = {num_accepted_samples}")
        )
        print(format_line(f"Number of total samples = {num_total_samples}"))
        print("-" * line_length)
        print(format_line(f"Acceptance rate = {acceptance_rate:.2f} %"))
        print("-" * line_length)


class Minimizing1d:
    def __init__(self, method: str):
        assert method in [
            "golden section",
            "brent method",
        ], "Method must be either 'golden section' or 'brent method'"
        self.method = method

        self.golden_ratio = 1.61803398875

        self.w = 2 - self.golden_ratio

    def minimize(self, func, a, b, tol=1e-5, viz=True, vizstep=5):
        if self.method == "golden section":
            return self._golden_section(func, a, b, tol, viz, vizstep)
        elif self.method == "brent method":
            return self._brent_method(func, a, b, tol, viz, vizstep)

    def _brent_method(self, func, a, b, tol, viz, vizstep):
        if viz:
            print("Brent's method vizualization not implemented yet!")
            print("Proceeding without vizualization...")

        max_num_steps = 100  # Hardcoded (maybe change later)

        a, b, c = self._bracketing(func, a, b)

        w = b
        v = b
        count = 0
        last_step = np.inf
        while abs(c - a) > 2 * b * tol or count < max_num_steps:
            if w == b and v == b:
                # First step: golden section step
                a, b, c, d, _ = self._golden_step(a, b, c)
            else:
                d = self._parabolic_interpolation(
                    func,
                    b,
                )

            current_step = abs(d - b)
            if not a < d < c and current_step < last_step:
                a, b, c, d, _ = self._golden_step(a, b, c)

    def _golden_section(self, func, a, b, tol, viz, vizstep):
        a, b, c = self._bracketing(func, a, b)

        old_x = None
        count = 0

        xx = np.linspace(0, 5, 100)
        func_vals = func(xx)
        if viz:
            os.makedirs("MinimizationPlotting", exist_ok=True)
            plt.plot(xx, func_vals)
            plt.title(f"Golden section method - step {count} of -N(x) minimization")
            plt.axvline(a, color="r", linestyle="--")
            plt.axvline(b, color="g", linestyle="--")
            plt.axvline(c, color="b", linestyle="--")
            plt.savefig(f"MinimizationPlotting/golden_section_step_{count}.png")
            plt.close()
        while abs(c - a) > tol:
            count += 1
            if old_x is None:
                interval1 = abs(c - b)
                interval2 = abs(b - a)

                if interval1 < interval2:
                    x = a
                elif interval1 > interval2:
                    x = c
            else:
                if old_x == a:
                    x = c
                else:  # old_x == c
                    x = a

            a, b, c, d, old_x = self._golden_step(a, b, c, x)

            if viz and count % vizstep == 0:
                func_vals = func(xx)
                plt.plot(xx, func_vals)
                plt.title(f"Golden section method - step {count} of -N(x) minimization")
                plt.axvline(a, color="r", linestyle="--")
                plt.axvline(b, color="g", linestyle="--")
                plt.axvline(c, color="b", linestyle="--")
                plt.savefig(f"MinimizationPlotting/golden_section_step_{count}.png")
                plt.close()

        print(f"Golden section method took {count} iterations")
        if func(d) < func(b):
            return d
        else:
            return b

    def _golden_step(self, a, b, c, x=None):
        if x is None:
            interval1 = abs(c - b)
            interval2 = abs(b - a)

            if interval1 < interval2:
                x = a
            elif interval1 > interval2:
                x = c

        d = b + (x - b) * self.w

        if b < d < c:
            a = b
            b = d
        elif a < d < b:
            c = b
            b = d
        return a, b, c, d, x

    def _bracketing(self, func, a, b):
        # Is f(a) < f(b)? If not, swap a and b
        flip_at_the_end = False
        if not func(a) > func(b):
            a, b = b, a
            flip_at_the_end = True

        c_proposed = b + (b - a) * self.golden_ratio
        if func(c_proposed) > func(b):
            c = c_proposed

        while func(c_proposed) < func(b):
            # Fit a parabola through a, b and c_proposed and find the minimum of the parabola
            d = self._parabolic_interpolation(func, a, b, c_proposed)

            if d is None:
                # If the parabola is a bad fit, we just do a golden section step
                d = c_proposed + (c_proposed - b) * self.golden_ratio

            # If d is between b and c_proposed, we might be done
            if b < d < c_proposed:
                if func(d) < func(c_proposed):
                    # Bracket [b,d,c_proposed]
                    a = b
                    b = d
                    c = c_proposed
                elif func(d) > func(c_proposed):
                    # Bracket [a,b,d]:
                    c = d
                else:
                    # If neither is the case the parabola is a bad fit so we just do
                    d = c_proposed + (c_proposed - b) * self.golden_ratio
            if not b < d < c_proposed:
                # d is not between b and c_proposed
                # we check if it is too far away
                dist = abs(d - b)
                if dist > 100 * abs(c_proposed - b):
                    # If it is too far away, we just do a golden section step
                    d = c_proposed + (c_proposed - b) * self.golden_ratio

            if not b < d < c_proposed:
                a = b
                b = c_proposed
                c = d
            else:
                a = b
                b = d
                c = c_proposed
                break

        if flip_at_the_end:
            return c, b, a
        return a, b, c

    def _parabolic_interpolation(self, func, a, b, c):
        # Fit a parabola through a, b and c and find the minimum of the parabola
        fa = func(a)
        fb = func(b)
        fc = func(c)

        numerator = (b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)
        denominator = 2 * ((b - a) * (fb - fc) - (b - c) * (fb - fa))

        if denominator == 0:
            return None

        x_min = b - numerator / denominator
        return x_min


class LikelihoodMinimizer:
    def __init__(
        self,
        likelihood: callable,
        likelihood_derivative: callable,
        params_to_optimize: dict,
        params_not_to_optimize: dict,
        model: callable,
        data: np.ndarray,
        gamma: float = 1.0,
        verbose: bool = False,
        verbose_logging: bool = False,
    ):
        """
        Implementation of the Newton-Raphson method for root finding,
        specifically tailored for minimizing likelihood functions.

        P.S: I would have used autodiff for the derivative of the function which makes
        it more general but I was lazy to implement autodiff from scratch and we can't use
        external libraries ;(

        :param likelihood: The likelihood function for which we want to find the minimum.
        :param likelihood_kwargs: A dictionary of keyword arguments to pass to the likelihood function.
        :param likelihood_derivative: The derivative of the likelihood function.
        :param likelihood_derivative_kwargs: A dictionary of keyword arguments to pass to the derivative function.
        :param model: The model function that we are fitting to the data.
        :param gamma: A damping factor to control the step size (default is 1.0).
        :param verbose: Whether to print verbose output during the minimization process (default is False).
        :param verbose_logging: Whether to log detailed information about each iteration (default is False).
        """
        self._likelihood = likelihood
        self._likelihood_derivative = likelihood_derivative

        self._model = model

        self._data = data

        self._params_to_optimize = params_to_optimize
        self._params_not_to_optimize = params_not_to_optimize

        self._gamma = gamma

        self._verbose = verbose
        self._verbose_logging = verbose_logging

    @property
    def history(self):
        """
        Returns the history of guesses,
        function values, derivative values, and step sizes
        during the root finding process.
        """
        if hasattr(self, "_history"):
            return self._history
        else:
            print(
                "No history found. Please run estimate_root with data_logging=True to log the history of guesses."
            )
            return {}

    def minimize(
        self,
        initial_params: dict,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        maximum_iterations: int = 100,
        data_logging: bool = False,
        log_every_n_iterations: int = 1,
    ):
        """
        Minimize the chi-squared function using the Newton-Raphson method.
        The method iteratively updates the guess for the minimum using the formula:
            new_guess = guess - gamma * likelihood(guess) / likelihood'(guess)
        The iteration continues until either the absolute error or the
        relative error is less than the specified tolerance.

        :param initial_params: Initial guess for the minimum as a dictionary of parameter values.
        :param atol: Absolute tolerance for convergence (default is 1e-6).
        :param rtol: Relative tolerance for convergence (default is 1e-6).
        :param maximum_iterations: Maximum number of iterations to perform (default is 100).
        :param data_logging: Whether to log the history of guesses,
        function values, derivative values, and step sizes (default is False).
        :param log_every_n_iterations: Log data every n iterations if data_logging is True (default is 1).
        :return: A tuple containing the estimated minimum, absolute error, and relative error.
        """
        if set(initial_params.keys()) != set(self._params_to_optimize.keys()):
            raise ValueError(
                "Initial parameters must have the same keys as params_to_optimize."
            )

        initial_guess_arr = self._params_to_arr(initial_params)

        iteration_count = 0
        self._history = {}
        while True:
            # Break if the maximum number of iterations is reached
            if iteration_count >= maximum_iterations:
                print("Maximum number of iterations reached without convergence.")
                break
            iteration_count += 1

            current_guess_dict = self._arr_to_full_params(initial_guess_arr)
            current_likelihood = self._likelihood(
                model=self._model, data=self._data, params=current_guess_dict
            )

            trial_gamma = self._gamma / np.sqrt(iteration_count)
            accepted_step = False
            for _ in range(20):
                proposed_guess_arr = initial_guess_arr - self._step(
                    initial_guess_arr, trial_gamma
                )
                new_guess_arr = self._project_params(proposed_guess_arr)
                new_guess_dict = self._arr_to_full_params(new_guess_arr)
                new_likelihood = self._likelihood(
                    model=self._model, data=self._data, params=new_guess_dict
                )

                if np.isfinite(new_likelihood) and (
                    not np.isfinite(current_likelihood) or new_likelihood <= current_likelihood
                ):
                    accepted_step = True
                    break
                trial_gamma *= 0.5

            if not accepted_step:
                print("Unable to find a chi-squared lowering step.")
                break

            # Calculating the difference between each guess (absolute error)
            aerr = (np.abs(new_guess_arr - initial_guess_arr)).max()

            # Calculating the relative error
            rerr = aerr / np.abs(new_guess_arr).max()

            likelihood_change = abs(new_likelihood - current_likelihood)

            # Change the initial guess to the current guess
            initial_guess_arr = new_guess_arr

            if data_logging:
                if iteration_count % log_every_n_iterations == 0:
                    self._history = self._logging(
                        initial_guess_arr,
                        history=self._history,
                        itteration_count=iteration_count,
                    )

            # Break if either the absolute error or the
            # relative error is less than the threshold
            if self._check_convergence(aerr, rerr, atol, rtol, iteration_count):
                break
            if likelihood_change < atol or likelihood_change < rtol * max(abs(current_likelihood), 1.0):
                if self._verbose:
                    print(f"Likelihood minimization converged at iteration {iteration_count}!")
                break

        final_params = self._arr_to_params(initial_guess_arr)
        for params in self._params_not_to_optimize:
            final_params[params] = self._params_not_to_optimize[params]
        return final_params, aerr, rerr

    def _arr_to_full_params(self, arr: np.ndarray) -> dict:
        params = self._arr_to_params(arr)
        for param in self._params_not_to_optimize:
            params[param] = self._params_not_to_optimize[param]
        return params

    def _params_to_arr(self, params: dict) -> np.ndarray:
        """
        Convert a dictionary of parameter values to a numpy array in the order of self._params_to_optimize.

        :param params: A dictionary of parameter values.
        :return: A numpy array of parameter values in the order of self._params_to_optimize.
        """
        return np.array([params[param] for param in self._params_to_optimize])

    def _arr_to_params(self, arr: np.ndarray) -> dict:
        """
        Convert a numpy array of parameter values to a dictionary in the order of self._params_to_optimize.

        :param arr: A numpy array of parameter values in the order of self._params_to_optimize.
        :return: A dictionary of parameter values.
        """
        return {param: arr[i] for i, param in enumerate(self._params_to_optimize)}

    def plot_history(self, filename="chi_sqrt_minimization_history.png"):
        """
        Plot the history of guesses, function values,
        derivative values, and step sizes during the root finding process.
        """
        import matplotlib.pyplot as plt

        self._history = self.history

        iterations = self._history["iteration"]
        guesses = self._history["guess"]
        function_values = self._history["function_value"]
        derivative_values = self._history["derivative_value"]
        step_sizes = self._history["step_size"]

        fig, axs = plt.subplots(2, 2, figsize=(10, 20))

        axs[0, 0].plot(iterations, guesses, marker="o")
        axs[0, 0].set_title("Guess vs Iteration")
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Guess")

        axs[0, 1].plot(iterations, function_values, marker="o")
        axs[0, 1].set_title("Function Value at Guess vs Iteration")
        axs[0, 1].set_xlabel("Iteration")
        axs[0, 1].set_ylabel("Function Value at Guess")

        axs[1, 0].plot(iterations, derivative_values, marker="o")
        axs[1, 0].set_title("Derivative Value at Guess vs Iteration")
        axs[1, 0].set_xlabel("Iteration")
        axs[1, 0].set_ylabel("Derivative Value at Guess")

        axs[1, 1].plot(iterations, step_sizes, marker="o")
        axs[1, 1].set_title("Step Size vs Iteration")
        axs[1, 1].set_xlabel("Iteration")
        axs[1, 1].set_ylabel("Step Size")

        plt.savefig(f"Plots/{filename}")
        plt.close()

    def _step(self, guess_arr: np.ndarray, gamma: float) -> np.ndarray:
        """
        Calculate the step size for the Newton-Raphson method.
        The step size is given by the chi-squared value at the current guess
        divided by the derivative of the chi-squared value at the current guess, multiplied by the damping factor gamma.
            step = gamma * likelihood(guess) / likelihood'(guess)

        :param guess_arr: The current guess for the minimum as a numpy array.
        :param gamma: The damping factor for the step size.
        :return: The step size to update the guess.
        """
        guess_dict = self._arr_to_full_params(guess_arr)
        gradient = np.asarray(
            self._likelihood_derivative(
                model=self._model, data=self._data, params=guess_dict
            ),
            dtype=float,
        )

        if gradient.shape != guess_arr.shape or not np.all(np.isfinite(gradient)):
            return np.zeros_like(guess_arr)

        step = gamma * gradient
        return np.clip(step, -0.2, 0.2)

    def _project_params(self, guess_arr: np.ndarray) -> np.ndarray:
        projected = np.array(guess_arr, dtype=float, copy=True)
        for index, param in enumerate(self._params_to_optimize):
            if param == "a":
                projected[index] = np.clip(projected[index], 0.5, 10.0)
            elif param == "b":
                projected[index] = np.clip(projected[index], 1e-3, X_MAX)
            elif param == "c":
                projected[index] = np.clip(projected[index], 0.1, 10.0)
        return projected

    def _check_convergence(
        self, aerr: float, rerr: float, atol: float, rtol: float, iteration_count: int
    ) -> bool:
        """
        Check if the convergence criteria are met based on the absolute error and relative error.

        :param aerr: The absolute error of the current guess.
        :param rerr: The relative error of the current guess.
        :param atol: The absolute tolerance for convergence.
        :param rtol: The relative tolerance for convergence.
        :param iteration_count: The current iteration count.
        :return: True if convergence is successful, False otherwise.
        """
        if aerr < atol or rerr < rtol:
            print(f"Convergence successful at iteration {iteration_count}!")
            return True
        return False

    def _logging(
        self, guess_arr: np.ndarray, history: dict = {}, itteration_count: int = 0
    ):
        func_value = self._likelihood(
            model=self._model, data=self._data, params=self._arr_to_full_params(guess_arr)
        )
        derivative_falue = np.asarray(
            self._likelihood_derivative(
                model=self._model, data=self._data, params=self._arr_to_full_params(guess_arr)
            ),
            dtype=float,
        )
        step_size = self._step(guess_arr, self._gamma)

        # Initialize the history dictionary if it doesn't exist
        if "iteration" not in history:
            history["iteration"] = []
        if "guess" not in history:
            history["guess"] = []
        if "function_value" not in history:
            history["function_value"] = []
        if "derivative_value" not in history:
            history["derivative_value"] = []
        if "step_size" not in history:
            history["step_size"] = []

        # Append the current iteration data to the history
        history["iteration"].append(itteration_count)
        history["guess"].append(guess_arr)
        history["function_value"].append(func_value)
        history["derivative_value"].append(derivative_falue)
        history["step_size"].append(step_size)

        if self._verbose_logging:
            print("-----------------------------")
            print(f"Iteration {itteration_count}:")
            print(f"Current guess: {guess_arr}")
            print(f"Function value at current guess: {func_value}")
            print(f"Derivative value at current guess: {derivative_falue}")
            print(f"Step size: {step_size}")
            print("-----------------------------")
        return history


def finite_differences_gradient(likelihood_fn: callable, model: callable, data: np.ndarray, params: dict) -> float:
    """
    Computes the gradient of the likelihood function with respect to the parameters a, b, and c
    using finite differences.
    """
    gradients = []
    for parameter in ("a", "b", "c"):
        step = max(1e-4, 1e-4 * abs(params[parameter]))
        params_plus = dict(params)
        params_minus = dict(params)
        params_plus[parameter] += step
        params_minus[parameter] -= step

        likelihood_plus = likelihood_fn(model, data, params_plus)
        likelihood_minus = likelihood_fn(model, data, params_minus)

        if not np.isfinite(likelihood_plus) and not np.isfinite(likelihood_minus):
            gradients.append(0.0)
        elif not np.isfinite(likelihood_minus):
            gradients.append((likelihood_plus - likelihood_fn(model, data, params)) / step)
        elif not np.isfinite(likelihood_plus):
            gradients.append((likelihood_fn(model, data, params) - likelihood_minus) / step)
        else:
            gradients.append((likelihood_plus - likelihood_minus) / (2 * step))

    return np.array(gradients)

def minimize_likelihood(
        likelihood_fn: callable,
        likelihood_derivative_fn: callable,
        model: callable,
        data: np.ndarray,
        initial_params: tuple,
        datafile: str,
        likelihood_fn_name: str = "chi2",
        plot: bool = True,
        verbose: bool = True
    ) -> tuple:
    minimzer = LikelihoodMinimizer(
        likelihood=likelihood_fn,
        likelihood_derivative=likelihood_derivative_fn,
        params_to_optimize={
            "a": initial_params[0],
            "b": initial_params[1],
            "c": initial_params[2],
        },
        params_not_to_optimize={"Nsat": data["Nsat"]},
        model=model,
        data=data,
        gamma=0.01,
        verbose=verbose,
    )

    final_params, _, _ = minimzer.minimize(
        initial_params={
            "a": initial_params[0],
            "b": initial_params[1],
            "c": initial_params[2],
        },
        atol=1e-4,
        rtol=1e-4,
        maximum_iterations=500,
        data_logging=True,
    )

    if plot:
        minimzer.plot_history(filename=f"{likelihood_fn_name}_minimization_history_{datafile}.png")

    best_params = (final_params["a"], final_params["b"], final_params["c"])
    min_likelihood = likelihood_fn(
        model,
        data,
        {
            "a": best_params[0],
            "b": best_params[1],
            "c": best_params[2],
            "Nsat": data["Nsat"],
        },
    )
    return best_params, min_likelihood


def G_test(observed: np.ndarray, expected: np.ndarray) -> float:
    """
    Perform the G-test for goodness of fit.

    Parameters
    ----------
    observed : ndarray
        The observed counts in each bin.
    expected : ndarray
        The expected counts in each bin according to the model.
    Returns
    -------
    float
        The G-test statistic.
    """
    # No way to compare if there are no observed counts
    # Extra precaution to avoid log(0) which shouldn't happen
    if len(observed) == 0:
        return np.inf
    
    mask = (observed > 0) & (expected > 0) # Only consider valid bins
    observed = observed[mask] 
    expected = expected[mask]

    return 2 * np.sum(observed * np.log(observed / expected))


def chi2_cdf(x: float, dof: float) -> float:
    """
    Compute P(chi2 <= x) where chi2 follows chi-squared distribution with dof degrees of freedom.
    Uses the relationship: P(chi2_k <= x) = P(gamma(k/2) <= x/2) where P is regularized lower incomplete gamma.
    
    Parameters
    ----------
    x : float
        Chi-squared statistic value
    dof : float
        Degrees of freedom
    
    Returns
    -------
    float
        CDF value in [0, 1]
    """
    if x <= 0:
        return 0.0
    if dof <= 0:
        return 1.0
    
    a = dof / 2.0  #shape parameter for gamma
    z = x / 2.0    # scale parameter
    
    # Compute regularized lower incomplete gamma: P(a, z) = gamma(a, z) / Gamma(a)
    # Using series expansion for P(a, z)
    # P(a, z) = (1/Gamma(a)) * ∫_0^z t^(a-1) * e^(-t) dt
    
    # Compute using series: sum_{k=0}^inf (-1)^k * z^k / k! / (a + k)
    # This is equivalent to: P(a, z) = e^(-z) * z^a / Gamma(a) * sum_{k=0}^inf z^k / Gamma(a+k+1)
    
    try:
        # Get Gamma(a)
        gamma_a = gamma(a)
    except (ValueError, OverflowError):
        # Fallback if gamma computation fails
        return min(1.0, x / (x + dof))
    
    # Series expansion with careful numerics
    result = 0.0
    term = 1.0 / gamma_a  # First coefficient
    
    for k in range(300):
        # Add term to sum
        result += term
        
        # Compute next term: term_{k+1} = term_k * z / (a + k + 1)
        term *= z / (a + k + 1)
        
        # Stop if term is negligibly small
        if abs(term) < 1e-15 * abs(result):
            break
    
    # Multiply by e^(-z) * z^a to get the actual integral
    try:
        if z > 100:  # Avoid exponential underflow
            log_factor = -z + a * np.log(z) - np.log(gamma_a)
            factor = np.exp(log_factor)
        else:
            factor = np.exp(-z) * (z ** a) / gamma_a
        
        result *= factor
    except (ValueError, OverflowError):
        pass
    
    return min(1.0, max(0.0, result))


def Q_test(observed: np.ndarray, expected: np.ndarray, dof: int = None) -> float:
    """
    Compute the p-value (significance Q) for the goodness of fit test.
    
    This computes the chi-squared test statistic and converts it to a p-value
    using the chi-squared CDF.

    Parameters
    ----------
    observed : ndarray
        The observed counts in each bin.
    expected : ndarray
        The expected counts in each bin according to the model.
    dof : int, optional
        Degrees of freedom. If None, defaults to number of bins - 3 (for 3 fitted parameters).
    
    Returns
    -------
    float
        The p-value (Q): P(chi2 >= observed_statistic)
    """
    # No way to compare if there are no observed counts
    if len(observed) == 0:
        return np.inf
    
    mask = (observed > 0) & (expected > 0) # Only consider valid bins
    observed = observed[mask] 
    expected = expected[mask]
    
    # Compute chi-squared test statistic
    chi2_stat = np.sum((observed - expected) ** 2 / expected)
    
    # If dof not provided, use number of bins minus 3 (for 3 fitted parameters a, b, c)
    if dof is None:
        dof = len(observed) - 3
    
    # Ensure dof is at least 1
    if dof <= 0:
        dof = 1
    
    # Compute p-value: Q = P(chi2 > chi2_stat) = 1 - CDF(chi2_stat)
    cdf_value = chi2_cdf(chi2_stat, dof)
    Q = 1.0 - cdf_value
    
    return Q


def get_best_params_for_datafile(datafile: str, table_name: str) -> tuple:
    base_path = "Calculations"
    table_path = os.path.join(base_path, table_name)

    # Throw an error if the table doesn't exist
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"Table {table_name} not found in {base_path} directory. Please run . run.sh to generate it!")
    
    # Read from .tex table
    # Lines look like f"m{idx+11} & {N:.5f} & {chi2_val:.5f} & {a:.5f} & {b:.5f} & {c:.5f}{line_end}\n"
    params = {
        "N": None,
        "a": None,
        "b": None,
        "c": None,
    }
    with open(table_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(datafile):
                parts = line.split("&")
                params["N"] = float(parts[1].strip())
                params["a"] = float(parts[3].strip())
                params["b"] = float(parts[4].strip())
                params["c"] = float(parts[5].strip().rstrip("\\"))
                return params


def mcmc_proposal_normal(current_sample: np.ndarray, sigma: float) -> np.ndarray:
    """
    Gaussian random-walk proposal for the 1D MCMC chain.
    Returns shape (1, 1) to match the MCMC class' indexing convention.
    """
    current = float(np.atleast_1d(current_sample)[0])
    proposed = np.random.normal(loc=current, scale=sigma)
    return np.array([[proposed]])


def sample_radii_with_mcmc(
    params: dict,
    num_samples: int,
    x_lower: float = 1e-4,
    x_upper: float = X_MAX,
    burn_in: int = 1000,
    proposal_sigma: float = 0.08,
) -> np.ndarray:
    """
    Draw satellite radii samples from the shell-count profile using Metropolis-Hastings.
    """
    A = get_normalization_constant(params["a"], params["b"], params["c"], params["Nsat"])

    def target_density(parameters: np.ndarray, data: tuple[np.ndarray, np.ndarray]) -> float:
        x = float(np.atleast_1d(parameters)[0])
        if x <= x_lower or x >= x_upper:
            return 1e-300
        val = N(x, A, params["Nsat"], params["a"], params["b"], params["c"])
        if not np.isfinite(val) or val <= 0:
            return 1e-300
        return float(val)

    sampler = MCMC(
        likelihood=target_density,
        proposed_distribution=mcmc_proposal_normal,
        sigma=proposal_sigma,
    )

    # Start chain near the characteristic scale b, clipped to the allowed domain.
    initial_x = np.clip(params["b"], x_lower * 1.2, x_upper * 0.8)
    retained_samples = sampler.metropolis_hastings(
        initial_sample=np.array([initial_x], dtype=float),
        data=(np.array([]), np.array([])),
        num=num_samples + burn_in,
        burn_in=burn_in,
    )

    radii = np.array([float(np.atleast_1d(s)[0]) for s in retained_samples], dtype=float)
    return radii[:num_samples]


# =====================================================
# =====================================================
# =====================================================


# =====================================================
# ============= Wrappers from template ================
# =====================================================


def readfile(filename):
    """
    Helper function to read in the satellite galaxy data from the provided text files.

    Parameters
    ----------
    filename : str
        The name of the file to read in.

    Returns
    -------
    radius : ndarray
        The virial radius for all the satellites in the file.
    nhalo : int
        The number of halos in the file.
    """
    f = open(filename, "r")
    data = f.readlines()[3:]  # Skip first 3 lines
    nhalo = int(data[0])  # number of halos
    radius = []

    for line in data[1:]:
        if line[:-1] != "#":
            radius.append(float(line.split()[0]))

    radius = np.array(radius, dtype=float)
    f.close()
    return (
        radius,
        nhalo,
    )  # Return the virial radius for all the satellites in the file, and the number of halos


def n(x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float) -> np.ndarray:
    """
    Number density profile of satellite galaxies

    Parameters
    ----------
    x : ndarray
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    ndarray
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    b_inverse = 1 / b
    return A * Nsat * (x * b_inverse) ** (a - 3) * np.exp(-((x * b_inverse) ** c))


def N(x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float) -> np.ndarray:
    """
    N(x)dx = n(x) * 4 * pi * x^2 dx is the number of satellite galaxies in a shell of radius x and thickness dx.
    """
    return n(x, A, Nsat, a, b, c) * 4 * np.pi * x**2


def dN_da(
    x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> np.ndarray:
    """
    Partial derivative of N(x) with respect to a.
    """
    return dn_da(x, A, Nsat, a, b, c) * 4 * np.pi * x**2


def dN_db(
    x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> np.ndarray:
    """
    Partial derivative of N(x) with respect to b.
    """
    return dn_db(x, A, Nsat, a, b, c) * 4 * np.pi * x**2


def dN_dc(
    x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> np.ndarray:
    """
    Partial derivative of N(x) with respect to c.
    """
    return dn_dc(x, A, Nsat, a, b, c) * 4 * np.pi * x**2


def dn_da(
    x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> np.ndarray:
    """
    Partial derivative of the number density profile of satellite galaxies with respect to a.
    """
    b_inverse = 1 / b
    return (
        A
        * Nsat
        * (x * b_inverse) ** (a - 3)
        * np.exp(-((x * b_inverse) ** c))
        * np.log(x * b_inverse)
    )


def dn_db(
    x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> np.ndarray:
    """
    Partial derivative of the number density profile of satellite galaxies with respect to b.
    """
    b_inverse = 1 / b
    u = x * b_inverse
    n_val = A * Nsat * u ** (a - 3) * np.exp(-(u**c))
    return n_val * b_inverse * ((3 - a) + c * u**c)


def dn_dc(
    x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> np.ndarray:
    """
    Partial derivative of the number density profile of satellite galaxies with respect to c.
    """
    b_inverse = 1 / b
    return (
        A
        * Nsat
        * (x * b_inverse) ** (a - 3)
        * np.exp(-((x * b_inverse) ** c))
        * (-((x * b_inverse) ** c))
        * np.log(x * b_inverse)
    )


# Following the lectures, the function below provides a template for a custom minimization method.
# Depending on your choice of method, you may or may not need to add more function input parameters.
def my_minimizer(
    func: callable,
    x_arr: np.ndarray,
    bounds: tuple,
    tol: float = 1e-5,
    method: str = "golden section",
) -> tuple:
    """
    Custom minimization method.

    Parameters
    ----------
    func : callable
        Function to minimize.
    x_arr : ndarray
        Array of x values to evaluate func at.
    bounds : tuple
        Tuple of (xmin, xmax) to search for minimum in.
    tol : float, optional
        Tolerance for the minimization.
        The default is 1e-5.

    Returns
    -------
    x_min : float
        Value of x at which func is minimum.
    func_min : float
        Minimum value of func.
    """
    minimizer = Minimizing1d(method)
    x_min = minimizer.minimize(func, bounds[0], bounds[1], tol)
    func_min = func(x_min)
    return x_min, func_min


def integrate_midpoint(
    func: callable, x_lower: float, x_upper: float, num_steps: int = 512
) -> float:
    dx = (x_upper - x_lower) / num_steps
    x_mid = x_lower + (np.arange(num_steps) + 0.5) * dx
    return np.sum(func(x_mid)) * dx


def build_binned_dataset(
    radius: np.ndarray,
    nhalo: int,
    x_lower: float,
    x_upper: float,
    bins: int,
) -> dict:
    bin_edges = np.logspace(np.log10(x_lower), np.log10(x_upper), bins + 1)
    counts, _ = np.histogram(radius, bins=bin_edges)
    mean_counts = counts / nhalo
    x_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    return {
        "bin_edges": bin_edges,
        "counts": mean_counts,
        "x_centers": x_centers,
        "Nsat": len(radius) / nhalo,
    }


def model_bin_means(bin_edges: np.ndarray, params: dict) -> np.ndarray:
    a = params["a"]
    b = params["b"]
    c = params["c"]
    Nsat = params["Nsat"]
    A = get_normalization_constant(a, b, c, Nsat)

    expected = []
    for x_left, x_right in zip(bin_edges[:-1], bin_edges[1:]):
        expected.append(
            integrate_midpoint(
                lambda x: N(x, A, Nsat, a, b, c),
                x_left,
                x_right,
                num_steps=256,
            )
        )
    return np.array(expected)


# Computes the model predictions for the given x values and parameters.
def get_plot_profile(x_values: np.ndarray, params: dict) -> np.ndarray:
    A = get_normalization_constant(
        params["a"], params["b"], params["c"], params["Nsat"]
    )
    return N(x_values, A, params["Nsat"], params["a"], params["b"], params["c"])


def is_valid_profile_params(params: dict) -> bool:
    a = params["a"]
    b = params["b"]
    c = params["c"]
    return 0.5 <= a <= 10.0 and 1e-3 <= b <= X_MAX and 0.1 <= c <= 10.0


#### Fitting ####


def chi2(model: callable, data: np.ndarray, params: dict) -> float:
    """
    Calculate the chi-squared for a given set of parameters and data.

    Parameters
    ----------
    model : callable
        The model function to compare to the data.
    data : ndarray
        The observed data to compare the model to.
    params : dict
        The parameters to evaluate the model at as a dictionary.

    Returns
    -------
    float
        The chi-squared value for the given parameters and data.
    """
    if not is_valid_profile_params(params):
        return np.inf

    expected = model(data["bin_edges"], params)
    observed = data["counts"]

    if np.any(~np.isfinite(expected)) or np.any(expected <= 0):
        return np.inf

    return np.sum((observed - expected) ** 2 / expected)


def chi2_partial_derivative(model: callable, data: np.ndarray, params: dict) -> float:
    """
    Computes the gradient of the chi-squared function with respect to the parameters a, b, and c
    using finite differences.
    """
    return finite_differences_gradient(chi2, model, data, params)

def negative_poisson_ln_likelihood(
    model: callable, data: np.ndarray, params: dict
) -> float:
    """
    Calculate the Poisson negative log-likelihood for a given set of parameters and data.

    Parameters
    ----------
    model : callable
        The model function to compare to the data.
    data : ndarray
        The observed data to compare the model to.
    params : dict
        The parameters to evaluate the model at as a dictionary.

    Returns
    -------
    float
        The Poisson negative log-likelihood value for the given parameters and data.
    """
    if not is_valid_profile_params(params):
        return np.inf

    expected = model(data["bin_edges"], params)
    observed = data["counts"]

    if np.any(~np.isfinite(expected)) or np.any(expected <= 0):
        return np.inf

    # Drop ln(N_i!) because it is constant with respect to model parameters.
    return np.sum(expected - observed * np.log(expected))

def negative_poisson_ln_likelihood_partial_derivative(model: callable, data: np.ndarray, params: dict) -> np.ndarray:
    """
    Computes the gradient of the Poisson negative log-likelihood function with respect to the parameters a, b, and c
    using finite differences.
    """
    return finite_differences_gradient(negative_poisson_ln_likelihood, model, data, params)

def get_normalization_constant(a: float, b: float, c: float, Nsat: float) -> float:
    """
    Calculate the normalization constant A (which is a function of a,b,c) for the satellite number density profile.

    Parameters
    ----------
    a : float
        Small-scale slope.
    b : float
        Transition scale.
    c : float
        Steepness of exponential drop-off.
    Nsat : float
        Average number of satellites.

    Returns
    -------
    float
        Normalization constant A.
    """
    if b <= 0 or c <= 0:
        return np.inf

    integral = integrate_midpoint(
        lambda x: 4 * np.pi * x**2 * (x / b) ** (a - 3) * np.exp(-((x / b) ** c)),
        1e-6,
        X_MAX,
        num_steps=4096,
    )

    if integral <= 0 or not np.isfinite(integral):
        return np.inf
    return 1 / integral


def minimize_chi2(
        model: callable,
        data: np.ndarray,
        initial_params: tuple,
        datafile: str,
        plot: bool = True,
        verbose: bool = True
    ) -> tuple:
    """
    Minimize the chi-squared value for a given model and data.

    Parameters
    ----------
    model : callable
        The model function to compare to the data.
    data : ndarray
        The observed data to compare the model to.
    initial_params : tuple
        Initial guess for the parameters to minimize over.

    Returns
    -------
    best_params : tuple
        The parameters that minimize the chi-squared value.
    min_chi2 : float
        The minimum chi-squared value achieved.
    """
    return minimize_likelihood(
        likelihood_fn=chi2,
        likelihood_derivative_fn=chi2_partial_derivative,
        model=model,
        data=data,
        initial_params=initial_params,
        datafile=datafile,       
        likelihood_fn_name="chi2",
        plot=plot,
        verbose=verbose
    )

def minimize_poisson_ln_likelihood(
    model: callable,
    data: np.ndarray,
    initial_params: tuple,
    datafile: str,
    plot: bool = True,
    verbose: bool = True
) -> tuple:
    """
    Minimize the Poisson negative log-likelihood for a given model and data.

    Parameters
    ----------
    model : callable
        The model function to compare to the data.
    data : ndarray
        The observed data to compare the model to.
    initial_params : tuple
        Initial guess for the parameters to minimize over.

    Returns
    -------
    best_params : tuple
        The parameters that minimize the Poisson negative log-likelihood value.
    min_ln_likelihood : float
        The minimum Poisson negative log-likelihood value achieved.
    """
    return minimize_likelihood(
        likelihood_fn=negative_poisson_ln_likelihood,
        likelihood_derivative_fn=negative_poisson_ln_likelihood_partial_derivative,
        model=model,
        data=data,
        initial_params=initial_params,
        datafile=datafile,
        likelihood_fn_name="poisson_ln_likelihood",
        plot=plot,
        verbose=verbose
    )

# =====================================================
# ======== Main functions for each subquestion ========
# =====================================================


def do_question_1a():
    # ======== Question 1a: Maximization of N(x) ========
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    A_1a = 256 / (5 * np.pi ** (3 / 2))
    x_lower, x_upper = 10**-4, 0.1

    N_x = lambda x: 4 * np.pi * x**2 * n(x, A_1a, Nsat, a, b, c)
    x_max, Nx_max = my_minimizer(
        lambda x: -N_x(x), np.array([0.0]), (x_lower, x_upper)
    )  # (-) because we want to maximize N(x)
    Nx_max = -Nx_max  # (-) because we minimized -N(x)

    print(f"Maximum of N(x) is {Nx_max:.6f} at x = {x_max:.6f}")

    # Write the results to text files for later use in the PDF
    with open("Calculations/satellite_max_x.txt", "w") as f:
        f.write(f"{x_max:.6f}")
    with open("Calculations/satellite_max_Nx.txt", "w") as f:
        f.write(f"{Nx_max:.6f}")


def do_question_1b():
    # ======== Question 1b: Fitting N(x) with chi-squared ========
    datafiles = ["m11", "m12", "m13", "m14", "m15"]

    N_sat = []
    min_chi2_values = []
    best_params_chi2 = []

    # initialize figure with 5 subplots on 3x2 grid for the 5 data files
    fig, axs = plt.subplots(3, 2, figsize=(6.4, 8.0))
    axs = axs.flatten()

    for datafile in datafiles:
        radius, nhalo = readfile(f"Data/satgals_{datafile}.txt")

        x_lower, x_upper = (
            10**-4,
            X_MAX,
        )
        bins = 15
        print("=============================================================================================")
        print(f"Fitting chi-squared for data file {datafile} with initial parameters a=2.4, b=0.25, c=1.6...")
        fit_data = build_binned_dataset(radius, nhalo, x_lower, x_upper, bins)
        best_params, min_chi2 = minimize_chi2(
            model=model_bin_means,
            data=fit_data,
            initial_params=(2.4, 0.25, 1.6),
            datafile=datafile,
        )
        print("=============================================================================================")

        Nsat_mean = fit_data["Nsat"]
        model_params = {
            "a": best_params[0],
            "b": best_params[1],
            "c": best_params[2],
            "Nsat": Nsat_mean,
        }
        model_counts = model_bin_means(fit_data["bin_edges"], model_params)

        # Store N_sat, chi2 values and best-fit parameters in their arrays
        N_sat.append(Nsat_mean)
        min_chi2_values.append(min_chi2)
        best_params_chi2.append(best_params)

        # Plot the data and the best-fit model for each data file in a subplot.
        ax = axs[datafiles.index(datafile)]
        ax.stairs(
            fit_data["counts"], fit_data["bin_edges"], color="tab:blue", label="Data"
        )
        ax.plot(
            fit_data["x_centers"], model_counts, color="tab:orange", label="Best fit"
        )

        # Add labels and title to the subplot
        ax.set_title(f"Data file: {datafile}")
        ax.set_xlabel("x = r / r_virial")
        ax.set_ylabel("Mean satellites / halo / bin")

        # log-log scaling
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=7)

    fig.delaxes(axs[-1])

    # Save the figure with all subplots
    plt.tight_layout()
    plt.savefig("Plots/satellite_fits_chi2.png")

    # Save N_sat, chi2 values and best-fit parameters for each data file to tex files for later use in the PDF
    with open("Calculations/table_fitparams_chi2.tex", "w") as f:
        rows = list(zip(N_sat, min_chi2_values, best_params_chi2))
        for idx, (N, chi2_val, params) in enumerate(rows):
            a, b, c = params
            line_end = " \\\\" if idx < len(rows) - 1 else ""
            f.write(
                f"m{idx+11} & {N:.5f} & {chi2_val:.5f} & {a:.5f} & {b:.5f} & {c:.5f}{line_end}\n"
            )


def do_question_1c():
    # ======== Question 1c: Fitting N(x) with Poisson ln-likelihood ========
    datafiles = ["m11", "m12", "m13", "m14", "m15"]

    N_sat = []
    min_poisson_llh_values = []
    best_params_poisson = []

    # initialize figure with 5 subplots on 3x2 grid for the 5 data files
    fig, axs = plt.subplots(3, 2, figsize=(6.4, 8.0))
    axs = axs.flatten()

    for datafile in datafiles:
        radius, nhalo = readfile(f"Data/satgals_{datafile}.txt")

        x_lower, x_upper = (
            10**-4,
            X_MAX,
        )
        bins = 15

        print("=============================================================================================")
        print(f"Fitting Poisson ln-likelihood for data file {datafile} with initial parameters a=2.4, b=0.25, c=1.6...")
        fit_data = build_binned_dataset(radius, nhalo, x_lower, x_upper, bins)
        best_params, min_poisson_llh = minimize_poisson_ln_likelihood(
            model=model_bin_means,
            data=fit_data,
            initial_params=(2.4, 0.25, 1.6),
            datafile=datafile,
        )
        print("=============================================================================================")

        Nsat_mean = fit_data["Nsat"]
        model_params = {
            "a": best_params[0],
            "b": best_params[1],
            "c": best_params[2],
            "Nsat": Nsat_mean,
        }
        model_counts = model_bin_means(fit_data["bin_edges"], model_params)

        # Store N_sat, Poisson NLL values and best-fit parameters in their arrays
        N_sat.append(Nsat_mean)
        min_poisson_llh_values.append(min_poisson_llh)
        best_params_poisson.append(best_params)

        # Plot the data and the best-fit model for each data file in a subplot.
        ax = axs[datafiles.index(datafile)]
        ax.stairs(
            fit_data["counts"], fit_data["bin_edges"], color="tab:blue", label="Data"
        )
        ax.plot(
            fit_data["x_centers"], model_counts, color="tab:orange", label="Best fit"
        )

        # Add labels and title to the subplot
        ax.set_title(f"Data file: {datafile}")
        ax.set_xlabel("x = r / r_virial")
        ax.set_ylabel("Mean satellites / halo / bin")

        # log-log scaling
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=7)

    fig.delaxes(axs[-1])

    # Save the figure with all subplots
    plt.tight_layout()
    plt.savefig("Plots/satellite_fits_poisson.png")

    # Save N_sat, Poisson NLL values and best-fit parameters for each data file to tex files for later use in the PDF
    with open("Calculations/table_fitparams_poisson.tex", "w") as f:
        rows = list(zip(N_sat, min_poisson_llh_values, best_params_poisson))
        for idx, (N, llh_val, params) in enumerate(rows):
            a, b, c = params
            line_end = " \\\\" if idx < len(rows) - 1 else ""
            f.write(
                f"m{idx+11} & {N:.5f} & {llh_val:.5f} & {a:.5f} & {b:.5f} & {c:.5f}{line_end}\n"
            )


def do_question_1d():
    # ======== Question 1d: Statistical tests ========
    datafiles = ["m11", "m12", "m13", "m14", "m15"]

    G_scores_chi2 = []
    Q_scores_chi2 = []

    G_scores_poisson = []
    Q_scores_poisson = []

    for datafile in datafiles:
        radius, nhalo = readfile(f"Data/satgals_{datafile}.txt")

        x_lower, x_upper = (
            10**-4,
            X_MAX,
        )
        bins = 15

        # Use best-fit parameters from previous steps
        best_params_chi2 = get_best_params_for_datafile(datafile, "table_fitparams_chi2.tex")
        best_params_poisson = get_best_params_for_datafile(datafile, "table_fitparams_poisson.tex")

        # Bin data
        fit_data = build_binned_dataset(radius, nhalo, x_lower, x_upper, bins)

        # Add Nsat to the best_params dictionaries for chi2 and poisson models
        best_params_chi2["Nsat"] = fit_data["Nsat"]
        best_params_poisson["Nsat"] = fit_data["Nsat"]

        # Compute model predictions for the binned data using the best-fit parameters for chi2 and poisson models
        chi2_model_counts = model_bin_means(fit_data["bin_edges"], best_params_chi2)
        poisson_model_counts = model_bin_means(fit_data["bin_edges"], best_params_poisson)

        # Calculate degrees of freedom: number of bins minus number of fitted parameters (a, b, c)
        dof = bins - 3
        
        # Append the G and Q scores for chi2 and poisson fits to their respective arrays
        G_scores_chi2.append(G_test(fit_data["counts"], chi2_model_counts))
        Q_scores_chi2.append(Q_test(fit_data["counts"], chi2_model_counts, dof=dof))
        G_scores_poisson.append(G_test(fit_data["counts"], poisson_model_counts))
        Q_scores_poisson.append(Q_test(fit_data["counts"], poisson_model_counts, dof=dof))

    # Save G and Q scores for chi2 and poisson fits to tex files for later use in the PDF
    with open("Calculations/statistical_test_table_rows.tex", "w") as f:
        rows = []
        for i, (G, Q) in enumerate(zip(G_scores_chi2, Q_scores_chi2), start=11):
            rows.append(f"$\\chi^2$ & m{i} & {G:.5f} & {Q:.5f}")

        for i, (G, Q) in enumerate(zip(G_scores_poisson, Q_scores_poisson), start=11):
            rows.append(f"Poisson & m{i} & {G:.5f} & {Q:.5f}")

        for idx, row in enumerate(rows):
            if idx < len(rows) - 1:
                f.write(row + " \\\\\n")
            else:
                f.write(row)


def do_question_1e():
    # ======== Question 1e: Monte Carlo simulations ========
    # pick one of the data files to perform the Monte Carlo simulations on, e.g. m12
    datafiles = ["m11", "m12", "m13", "m14", "m15"]
    index = (
        4  # index of the data file to use for Monte Carlo simulations, e.g. 1 for m12
    )
    datafile = datafiles[index]

    radius, nhalo = readfile(f"Data/satgals_{datafile}.txt")

    x_lower, x_upper = (1e-4, X_MAX)
    bins = 15
    real_fit_data = build_binned_dataset(radius, nhalo, x_lower, x_upper, bins)
    num_satellites = len(radius)

    # Use best-fit parameters from previous steps for the original data file
    best_params_chi2 = get_best_params_for_datafile(datafile, "table_fitparams_chi2.tex")
    best_params_poisson = get_best_params_for_datafile(
        datafile, "table_fitparams_poisson.tex"
    )

    original_params_chi2 = {
        "a": best_params_chi2["a"],
        "b": best_params_chi2["b"],
        "c": best_params_chi2["c"],
        "Nsat": real_fit_data["Nsat"],
    }
    original_params_poisson = {
        "a": best_params_poisson["a"],
        "b": best_params_poisson["b"],
        "c": best_params_poisson["c"],
        "Nsat": real_fit_data["Nsat"],
    }

    pseudo_chi2_params = []
    pseudo_poisson_params = []

    num_pseudo_experiments = 20
    for i in range(num_pseudo_experiments):
        # Draw new mock radii from each original best-fit model using MCMC.
        mock_radii_chi2 = sample_radii_with_mcmc(
            original_params_chi2,
            num_samples=num_satellites,
            x_lower=x_lower,
            x_upper=x_upper,
            burn_in=1200,
            proposal_sigma=0.08,
        )
        mock_radii_poisson = sample_radii_with_mcmc(
            original_params_poisson,
            num_samples=num_satellites,
            x_lower=x_lower,
            x_upper=x_upper,
            burn_in=1200,
            proposal_sigma=0.08,
        )

        chi2_mock_data = build_binned_dataset(
            mock_radii_chi2, nhalo, x_lower, x_upper, bins
        )
        poisson_mock_data = build_binned_dataset(
            mock_radii_poisson, nhalo, x_lower, x_upper, bins
        )

        best_chi2_params, _ = minimize_chi2(
            model=model_bin_means,
            data=chi2_mock_data,
            initial_params=(
                original_params_chi2["a"],
                original_params_chi2["b"],
                original_params_chi2["c"],
            ),
            datafile=f"mc_chi2_{datafile}_{i}",
            plot=False,
            verbose=False
        )
        best_poisson_params, _ = minimize_poisson_ln_likelihood(
            model=model_bin_means,
            data=poisson_mock_data,
            initial_params=(
                original_params_poisson["a"],
                original_params_poisson["b"],
                original_params_poisson["c"],
            ),
            datafile=f"mc_poisson_{datafile}_{i}",
            plot=False,
            verbose=False
        )

        pseudo_chi2_params.append(best_chi2_params)
        pseudo_poisson_params.append(best_poisson_params)

    # plot the pseudo best-fit profiles, plot the original best-fit profile in another color and plot the mean in one more color

    # chi2 plot
    x_plot = np.logspace(np.log10(x_lower), np.log10(x_upper), 300)
    plt.figure(figsize=(6.4, 4.8))
    for params in pseudo_chi2_params:
        pseudo_params = {
            "a": params[0],
            "b": params[1],
            "c": params[2],
            "Nsat": real_fit_data["Nsat"],
        }
        plt.plot(
            x_plot,
            get_plot_profile(x_plot, pseudo_params),
            color="tab:blue",
            alpha=0.20,
            linewidth=1.0,
        )

    plt.plot(
        x_plot,
        get_plot_profile(x_plot, original_params_chi2),
        color="tab:red",
        linewidth=2.0,
        label="Original fit",
    )

    mean_params_chi2 = np.mean(pseudo_chi2_params, axis=0)
    mean_profile_chi2 = get_plot_profile(
        x_plot,
        {
            "a": mean_params_chi2[0],
            "b": mean_params_chi2[1],
            "c": mean_params_chi2[2],
            "Nsat": real_fit_data["Nsat"],
        },
    )
    plt.plot(
        x_plot,
        mean_profile_chi2,
        color="black",
        linewidth=2.0,
        label="Mean pseudo fit",
    )

    plt.title(f"Monte Carlo simulations - chi2 fit - Data file: {datafiles[index]}")
    plt.xlabel("x = r / r_virial")
    plt.ylabel("Mean satellites / halo / bin")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/satellite_monte_carlo_chi2.png")
    plt.close()

    # poisson plot
    plt.figure(figsize=(6.4, 4.8))
    for params in pseudo_poisson_params:
        pseudo_params = {
            "a": params[0],
            "b": params[1],
            "c": params[2],
            "Nsat": real_fit_data["Nsat"],
        }
        plt.plot(
            x_plot,
            get_plot_profile(x_plot, pseudo_params),
            color="tab:blue",
            alpha=0.20,
            linewidth=1.0,
        )
    plt.plot(
        x_plot,
        get_plot_profile(x_plot, original_params_poisson),
        color="tab:red",
        linewidth=2.0,
        label="Original fit",
    )

    mean_params_poisson = np.mean(pseudo_poisson_params, axis=0)
    mean_profile_poisson = get_plot_profile(
        x_plot,
        {
            "a": mean_params_poisson[0],
            "b": mean_params_poisson[1],
            "c": mean_params_poisson[2],
            "Nsat": real_fit_data["Nsat"],
        },
    )
    plt.plot(
        x_plot,
        mean_profile_poisson,
        color="black",
        linewidth=2.0,
        label="Mean pseudo fit",
    )

    plt.title(f"Monte Carlo simulations - Poisson fit - Data file: {datafiles[index]}")
    plt.xlabel("x = r / r_virial")
    plt.ylabel("Mean satellites / halo / bin")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Plots/satellite_monte_carlo_poisson.png")
    plt.close()


if __name__ == "__main__":
    do_question_1a()
    do_question_1b()
    do_question_1c()
    do_question_1d()
    do_question_1e()
