"""Enhanced Dimroth-Watson distribution implementation.

Purpose
-------
This module implements the Dimroth-Watson angular distribution used to fit and
visualize alignment-angle histograms.

Provides
--------
- A scipy-compatible continuous distribution class.
- Probability density, cumulative distribution, sampling, and fitting helpers.
- Numerical utilities for stable concentration-parameter evaluation.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rv_continuous
from scipy.special import erf, erfi,erfc
from scipy.integrate import quad

from warnings import warn


__all__ = ('DimrothWatson')
__author__ = ('Duncan Campbell')


class DimrothWatson(rv_continuous):
    r"""
    A Dimroth-Watson distribution of :math:`\cos(\theta)`
    
    Parameters
    ----------
    k : float
        shape parameter
    
    Notes
    -----
    The Dimroth-Watson distribution is defined as:
    
    .. math::
        p(\cos(\theta)) = B(k)\exp[-k\cos(\theta)^2]\mathrm{d}\cos(\theta)
    
    where
    
    .. math::
        B(k) = \frac{1}{2}\int_0^1\exp(-k t^2)\mathrm{d}t
    
    We assume the ISO convention for spherical coordinates, where :math:`\theta`
    is the polar angle, bounded between :math:`[-\pi, \pi]`, and :math:`\phi`
    is the azimuthal angle, where for a Dimroth-Watson distribution, :math:`\phi`
    is a uniform random variable between :math:`[0, 2\pi]` for all `k`.
    
    For :math:`k<0`, the distribution of points on a sphere is bipolar.
    For :math:`k=0`, the distribution of points on a sphere is uniform.
    For :math:`k>0`, the distribution of points on a sphere is girdle.
    
    Note that as :math:`k \rarrow \infty`:
    
    .. math::
        p(\cos(\theta)) = \frac{1}{2}\left[ \delta(\cos(\theta) + 1) + \delta(\cos(\theta) - 1) \right]\mathrm{d}\cos(\theta)
    
    and as :math:`k \rarrow -\infty`:
    
    .. math::
        p(\cos(\theta)) = \frac{1}{2}\delta(\cos(\theta))\mathrm{d}\cos(\theta)
    
    For large :math:`|k|`, the attributes of this class are approximate and not well tested.
    """

    def _argcheck(self, k):
        r"""
        Check arguments
        """
        k = np.asarray(k)
        self.a = -1.0  # lower bound
        self.b = 1.0  # upper bound
        return (k == k)

    def _norm(self, k):
        r"""
        Normalization constant
        """
        k = np.atleast_1d(k)
        
        # Handle k=0 case
        if np.isclose(k, 0).all():
            return 0.5
        
        # Create an array to store the result
        norm = np.zeros_like(k)
        
        # For k>0
        positive_k = (k > 0)
        if np.any(positive_k):
            k_pos = k[positive_k]
            if np.any(k_pos > 100):  # Large positive k approximation
                large_mask = k_pos > 100
                k_large = k_pos[large_mask]
                norm[positive_k][large_mask] = np.sqrt(k_large / np.pi) / erf(np.sqrt(k_large))
                
                # Handle smaller positive k
                small_mask = ~large_mask
                if np.any(small_mask):
                    k_small = k_pos[small_mask]
                    integral = np.array([quad(lambda t: np.exp(-ki * t**2), 0, 1)[0] for ki in k_small])
                    norm[positive_k][small_mask] = 1 / (2 * integral)
            else:
                integral = np.array([quad(lambda t: np.exp(-ki * t**2), 0, 1)[0] for ki in k_pos])
                norm[positive_k] = 1 / (2 * integral)
        
        # For k<0
        negative_k = (k < 0)
        if np.any(negative_k):
            k_neg = -k[negative_k]  # Use absolute value for negative k
            if np.any(k_neg > 100):  # Large negative k approximation
                large_mask = k_neg > 100
                k_large = k_neg[large_mask]
                norm[negative_k][large_mask] = np.sqrt(k_large / np.pi) * np.exp(-k_large) / erfc(np.sqrt(k_large))
                
                # Handle smaller negative k
                small_mask = ~large_mask
                if np.any(small_mask):
                    k_small = k_neg[small_mask]
                    integral = np.array([quad(lambda t: np.exp(ki * t**2), 0, 1)[0] for ki in k_small])
                    norm[negative_k][small_mask] = 1 / (2 * integral)
            else:
                integral = np.array([quad(lambda t: np.exp(ki * t**2), 0, 1)[0] for ki in k_neg])
                norm[negative_k] = 1 / (2 * integral)
        
        return norm

    def _pdf(self, x, k):
        r"""
        Probability distribution function
        
        Parameters
        ----------
        k : float
            shape parameter
        
        Notes
        -----
        See the 'notes' section of the class for a discussion of large :math:`|k|`.
        """
        k = np.atleast_1d(k).astype(np.float64)
        x = np.atleast_1d(x).astype(np.float64)
        
        # Expand dimensions if necessary for broadcasting
        if k.ndim == 0:
            k = k[np.newaxis]
        if x.ndim == 0:
            x = x[np.newaxis]
        
        # Create output array
        p = np.zeros((len(k), len(x)))
        
        for i, ki in enumerate(k):
            norm = self._norm(ki)
            p[i] = norm * np.exp(-ki * x**2)
        
        # Handle edge cases
        epsilon = np.finfo(float).eps
        for i, ki in enumerate(k):
            # Large positive k (bipolar)
            if ki > 100:
                bipolar_mask = (x >= (1.0 - epsilon)) | (x <= (-1.0 + epsilon))
                p[i][bipolar_mask] = 1.0 / (2.0 * epsilon)
            
            # Large negative k (girdle)
            elif ki < -100:
                girdle_mask = (x >= (0.0 - epsilon)) & (x <= (0.0 + epsilon))
                p[i][girdle_mask] = 1.0 / (2.0 * epsilon)
        
        return p.squeeze()

    def log_likelihood(self, k, cos_theta):
        """
        Compute log-likelihood for given k and data
        
        Parameters
        ----------
        k : float
            Shape parameter
        cos_theta : array-like
            Array of cosθ values
            
        Returns
        -------
        float: Log-likelihood value
        """
        # Compute PDF values
        pdf_vals = self._pdf(cos_theta, k)
        
        # Avoid log(0) issues
        pdf_vals = np.clip(pdf_vals, 1e-12, None)
        
        return np.sum(np.log(pdf_vals))

    def fit(self, cos_theta, method='L-BFGS-B', bounds=(-500, 500), max_iter=5000):
        """
        Fit Dimroth-Watson distribution to cosθ data using maximum likelihood estimation
        
        Parameters
        ----------
        cos_theta : array-like
            Input cosθ values
        method : str, optional
            Optimization method to use (default: 'L-BFGS-B')
        bounds : tuple, optional
            Bounds for k parameter (default: (-50, 50))
        max_iter : int, optional
            Maximum number of iterations (default: 1000)
            
        Returns
        -------
        dict: Dictionary containing:
            'mu': Best-fit μ parameter
            'mu_error': Standard error of μ
            'kappa': Best-fit κ parameter
            'kappa_error': Standard error of κ
            'success': Optimization success flag
        """
        # Convert input to numpy array and validate
        cos_theta = np.asarray(cos_theta, dtype=float)
        valid_mask = np.isfinite(cos_theta) & (np.abs(cos_theta) <= 1)
        cos_theta = cos_theta[valid_mask]
        
        if len(cos_theta) == 0:
            raise ValueError("No valid cosθ data points")
        
        # Estimate initial k from data statistics
        mean_cos2 = np.mean(cos_theta**2)
        
        # Handle special cases first
        if np.isclose(mean_cos2, 1.0, atol=1e-6):  # All points at |cosθ|=1
            kappa_mle = 50.0 if np.mean(cos_theta) > 0 else -50.0
            mu_mle = -2 * np.arctan(kappa_mle) / np.pi
            return {
                'mu': mu_mle,
                'mu_error': 0.0,
                'kappa': kappa_mle,
                'kappa_error': 0.0,
                'success': True
            }
        
        # General initial estimate
        if mean_cos2 > 0.4:  # Highly aligned
            initial_k = 10.0
        elif mean_cos2 < 0.2:  # Highly anti-aligned
            initial_k = -10.0
        else:  # Moderate alignment
            initial_k = 5.0 * (mean_cos2 - 0.333) / 0.333
        
        # Constrain initial estimate
        initial_k = np.clip(initial_k, *bounds)
        
        # Negative log-likelihood function
        def neg_log_likelihood(k):
            return -self.log_likelihood(k, cos_theta)
        
        # Optimization
        result = minimize(
            neg_log_likelihood,
            x0=[initial_k],
            method=method,
            bounds=[bounds],
            options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        # Extract results
        success = result.success
        kappa_mle = result.x[0]
        mu_mle = -2 * np.arctan(kappa_mle) / np.pi
        
        # Error estimation using Fisher information
        n = len(cos_theta)
        if n > 10:  # Only compute errors with sufficient data
            try:
                # Numerical derivative for Fisher information
                dk = 0.01
                ll_plus = self.log_likelihood(kappa_mle + dk, cos_theta)
                ll_minus = self.log_likelihood(kappa_mle - dk, cos_theta)
                ll_center = self.log_likelihood(kappa_mle, cos_theta)
                
                # Second derivative approximation
                d2ll_dk2 = (ll_plus - 2*ll_center + ll_minus) / (dk**2)
                
                # Fisher information and standard error
                fisher_info = -d2ll_dk2
                kappa_error = 1.0 / np.sqrt(fisher_info) if fisher_info > 0 else 1.0/np.sqrt(n)
            except:
                kappa_error = 1.0 / np.sqrt(n)
        else:
            kappa_error = 1.0 / np.sqrt(n)
        
        # Error propagation for mu
        dmu_dk = -2 / (np.pi * (1 + kappa_mle**2))
        mu_error = np.abs(dmu_dk) * kappa_error
        
        return {
            'mu': mu_mle,
            'mu_error': mu_error,
            'kappa': kappa_mle,
            'kappa_error': kappa_error,
            'success': success
        }

    def _rvs(self, k, max_iter=100):
        r"""
        Random variate sampling
        
        Parameters
        ----------
        k : array_like
            array of shape parameters
        
        size : int, optional
            integer indicating the number of samples to draw.
            if not given, the number of samples will be equal to len(k).
        
        max_iter : int, optional
            integer indicating the maximum number of times to iteratively draw from
            the proposal distribution until len(s) points are accepted.
        
        Notes
        -----
        The random variate sampling for this distribution is an implementation
        of the rejection-sampling technique.
        
        The Proposal distributions are taken from Best & Fisher (1986).
        """
        k = np.atleast_1d(k).astype(np.float64)
        size = self._size[0]
        if size != 1:
            if len(k) == size:
                pass
            elif len(k) == 1:
                k = np.ones(size)*k
            else:
                msg = ('if `size` argument is given, len(k) must be 1 or equal to size.')
                raise ValueError(msg)
        else:
            size = len(k)
        
        # Vector to store random variates
        result = np.zeros(size)
        
        # Take care of k=0 case
        zero_k = (k == 0)
        uran0 = np.random.random(np.sum(zero_k))*2 - 1.0
        result[zero_k] = uran0
        
        # Take care of edge cases, i.e. |k| very large
        with np.errstate(over='ignore'):
            x = np.exp(k)
            inf_mask = np.array([False]*size)
        edge_mask = ((x == np.inf) | (x == 0.0))
        result[edge_mask & (k>0)] = np.random.choice([1,-1], size=np.sum(edge_mask & (k>0)))
        result[edge_mask & (k<0)] = 0.0
        
        # Apply rejection sampling technique to sample from pdf
        n_sucess = np.sum(zero_k) + np.sum(edge_mask)  # number of successful draws from pdf
        n_remaining = size - n_sucess  # remaining draws necessary
        n_iter = 0  # number of sample-reject iterations
        kk = k[(~zero_k) & (~edge_mask)]  # store subset of k values that still need to be sampled
        mask = np.array([False]*size)  # mask indicating which k values have a successful sample
        mask[zero_k] = True
        
        while (n_sucess < size) & (n_iter < max_iter):
            # Get three uniform random numbers
            uran1 = np.random.random(n_remaining)
            uran2 = np.random.random(n_remaining)
            uran3 = np.random.random(n_remaining)
            
            # Masks indicating which envelope function is used
            negative_k = (kk < 0.0)
            positive_k = (kk > 0.0)
            
            # Sample from g(x) to get y
            y = np.zeros(n_remaining)
            y[positive_k] = self.g1_isf(uran1[positive_k], kk[positive_k])
            y[negative_k] = self.g2_isf(uran1[negative_k], kk[negative_k])
            y[uran3 < 0.5] = -1.0*y[uran3 < 0.5]  # account for one-sided isf function
            
            # Calculate M*g(y)
            g_y = np.zeros(n_remaining)
            m = np.zeros(n_remaining)
            g_y[positive_k] = self.g1_pdf(y[positive_k], kk[positive_k])
            g_y[negative_k] = self.g2_pdf(y[negative_k], kk[negative_k])
            m[positive_k] = self.m1(kk[positive_k])
            m[negative_k] = self.m2(kk[negative_k])
            
            # Calculate f(y)
            f_y = self.pdf(y, kk)
            
            # Accept or reject y
            keep = ((f_y/(g_y*m)) > uran2)
            
            # Count the number of successful samples
            n_sucess += np.sum(keep)
            
            # Store y values
            result[~mask] = y
            
            # Update mask indicating which values need to be redrawn
            mask[~mask] = keep
            
            # Get subset of k values which need to be sampled.
            kk = kk[~keep]
            
            n_iter += 1
            n_remaining = np.sum(~keep)
        
        if (n_iter == max_iter):
            msg = ('The maximum number of iterations reached, random variates may not be representative.')
            warn(msg)
        
        return result

    def g1_pdf(self, x, k):
        r"""
        Proposal distribution for pdf for k>0
        """
        k = -1*k
        eta = np.sqrt(-1*k)
        C = eta/(np.arctan(eta))
        return (C/(1+eta2*x2))/2.0

    def g1_isf(self, y, k):
        r"""
        Inverse survival function of proposal distribution for pdf for k>0
        """
        k = -1*k
        eta = np.sqrt(-1*k)
        return (1.0/eta)*(np.tan(y*np.arctan(eta)))

    def m1(self, k):
        r"""
        Enveloping factor for proposal distribution for pdf for k>0
        """
        return 2.0*np.ones(len(k))

    def g2_pdf(self, x, k):
        r"""
        Proposal distribution for pdf for k<0
        """
        k = -1*k
        norm = 2.0*(np.exp(k)-1)/k
        return (np.exp(k*np.fabs(x)))/norm

    def g2_isf(self, y, k):
        r"""
        Inverse survival function of proposal distribution for pdf for k<0
        """
        k = -1.0*k
        C = k/(np.exp(k)-1.0)
        return np.log(k*y/C+1)/k

    def m2(self, k):
        r"""
        Enveloping factor for proposal distribution for pdf for k<0
        """
        k = -1.0*k
        C = k*(np.exp(k)-1)**(-1)
        norm = 2.0*(np.exp(k)-1)/k
        return C*norm
    def plot_fit(self, cos_theta, fit_result=None, symmetrize=False, 
                color_hist='blue', color_fit='red', color_band='red',
                alpha_hist=1.0, alpha_band=0.3, ci=90, param_name='kappa',
                hist_type='step', ax=None):
        """
        Plot histogram of cosθ data and fitted distribution with confidence interval
        
        Parameters
        ----------
        cos_theta : array-like
            Input cosθ values
        fit_result : dict, optional
            Result from fit() method. If not provided, will call fit() internally
        symmetrize : bool, optional
            If True, plot symmetric version (absolute values) of the distribution (default: False)
        color_hist : str, optional
            Color for histogram (default: 'blue')
        color_fit : str, optional
            Color for fitted curve (default: 'red')
        color_band : str, optional
            Color for confidence band (default: 'red')
        alpha_hist : float, optional
            Transparency for histogram (0-1, default: 1.0)
        alpha_band : float, optional
            Transparency for confidence band (0-1, default: 0.3)
        ci : float, optional
            Confidence interval percentage (0-100, default: 90)
        param_name : str, optional
            Parameter to use for density: 'kappa' or 'mu' (default: 'kappa')
        hist_type : str, optional
            Histogram type: 'step' shows only top line, 'bar' shows full bars (default: 'step')
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, create new figure
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object
        ax : matplotlib.axes.Axes
            The axes object
        
        Notes
        -----
        The confidence band is only plotted if fit_result contains bootstrap samples
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
        
        # Clean data
        cos_theta = np.asarray(cos_theta)
        valid_mask = np.isfinite(cos_theta) & (np.abs(cos_theta) <= 1)
        cos_theta = cos_theta[valid_mask]
        
        # Apply symmetrization if requested
        data = np.abs(cos_theta) if symmetrize else cos_theta
        
        # Fit data if no fit_result provided
        if fit_result is None:
            fit_result = self.fit(cos_theta, error_method='bootstrap' if ci is not None else 'MLE')
        
        # Histogram plot settings based on type
        hist_kwargs = {
            'density': True,
            'color': color_hist,
            'alpha': alpha_hist,
            'edgecolor': color_hist,
            'linewidth': 2
        }
        
        # Compute histogram
        bins = 40 if symmetrize else 40
        hist_range = [0, 1] if symmetrize else [-1, 1]
        
        # Plot histogram
        if hist_type == 'step':
            # Step histogram showing only top line
            counts, bin_edges = np.histogram(data, bins=bins, range=hist_range, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_width = bin_edges[1] - bin_edges[0]
            
            # Step plot with markers at bin centers
            ax.step(bin_edges, np.append(counts[0], counts), 
                   where='pre', color=color_hist, alpha=alpha_hist, linewidth=2)
        elif hist_type == 'bar':
            # Standard bar histogram
            ax.hist(data, bins=bins, range=hist_range, **hist_kwargs)
        else:
            raise ValueError("hist_type must be 'step' or 'bar'")
        
        # Generate points for fit curve
        x = np.linspace(hist_range[0], hist_range[1], 500)
        
        # Compute main fit curve
        y_fit = self.normalized_pdf(x, param_name=param_name, fit_result=fit_result)
        if symmetrize:
            # Account for symmetric PDF
            y_fit *= 2
        
        # Plot main fit curve
        ax.plot(x, y_fit, color=color_fit, linewidth=2, label=f'Best fit ({param_name}={fit_result[param_name]:.2f})')
        
        # Add confidence band if bootstrap samples are available
        bootstrap_samples = fit_result.get('bootstrap_kappa_samples' if param_name == 'kappa' 
                                          else 'bootstrap_mu_samples')
        
        if bootstrap_samples is not None and len(bootstrap_samples) > 0 and ci is not None:
            # Compute CI for each x value
            y_boot = np.zeros((len(bootstrap_samples), len(x)))
            
            for i, val in enumerate(bootstrap_samples):
                # Compute PDF for this parameter value
                y_boot[i] = self.normalized_pdf(x, param_name=param_name, param_value=val)
                if symmetrize:
                    y_boot[i] *= 2
            
            # Compute quantiles
            lower_bound = (100 - ci) / 2
            upper_bound = 100 - lower_bound
            
            y_low = np.percentile(y_boot, lower_bound, axis=0)
            y_high = np.percentile(y_boot, upper_bound, axis=0)
            
            # Plot confidence band
            ax.fill_between(x, y_low, y_high, color=color_band, alpha=alpha_band, 
                           label=f'{ci}% Confidence Band')
        
        # Add labels and title
        ax.set_xlabel(r'$|\cos\theta|$' if symmetrize else r'$\cos\theta$', fontsize=14)
        ax.set_ylabel('Probability Density', fontsize=14)
        
        title = 'Symmetrized ' if symmetrize else ''
        title += 'Dimroth-Watson Distribution Fit'
        ax.set_title(title, fontsize=16)
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Set symmetric x limits
        if not symmetrize:
            ax.set_xlim(-1.05, 1.05)
        
        plt.tight_layout()
        return fig, ax
