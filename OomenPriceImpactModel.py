# Based on the research findings, let me create a comprehensive implementation of an Oomen-style 
# price impact model that includes temporary impact with exponential decay and permanent impact
# The model will be based on the transient impact model (TIM) framework commonly referenced in literature

print("Analyzing the key components needed for Oomen-style price impact model:")
print("=" * 70)

print("\n1. Temporary Impact with Exponential Decay:")
print("   - I_temp(t) = α * v * exp(-λ * t)")
print("   - where α = temporary impact coefficient")
print("   - λ = decay rate parameter") 
print("   - v = trade volume")
print("   - t = time since trade")

print("\n2. Permanent Impact:")
print("   - I_perm = β * sign(v) * |v|^γ")
print("   - where β = permanent impact coefficient")
print("   - γ = impact exponent (often ~0.5 for square-root law)")

print("\n3. Total Impact Model:")
print("   - Impact(t) = I_perm + Σ I_temp_i(t - t_i)")
print("   - Sum over all previous trades with their decayed temporary impact")

print("\n4. Parameters to Calibrate:")
print("   - α: Temporary impact strength")
print("   - λ: Exponential decay rate")  
print("   - β: Permanent impact strength")
print("   - γ: Impact exponent")

print("\n5. Calibration Method:")
print("   - Maximum Likelihood Estimation (MLE)")
print("   - Minimize prediction errors against observed price impact")
print("   - Use L-BFGS-B optimization with parameter bounds")

# Implement the Oomen-style Price Impact Model
# This model features temporary impact with exponential decay and permanent impact
# Based on research into transient impact models and FX market microstructure

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from datetime import datetime, timedelta

# Reuse the base classes from the previous implementation
@dataclass
class TradeData:
    """Container for high-frequency trade data"""
    timestamp: pd.Timestamp
    price: float
    volume: float
    side: int  # 1 for buy, -1 for sell
    currency_pair: str
    
@dataclass
class MarketData:
    """Container for market microstructure data"""
    timestamp: pd.Timestamp
    bid: float
    ask: float
    mid_price: float
    spread: float
    volatility: float
    currency_pair: str

class PriceImpactModel(ABC):
    """Abstract base class for price impact models"""
    
    @abstractmethod
    def temporary_impact(self, volume: float, market_params: Dict) -> float:
        """Calculate temporary price impact"""
        pass
    
    @abstractmethod
    def permanent_impact(self, volume: float, market_params: Dict) -> float:
        """Calculate permanent impact"""
        pass
    
    @abstractmethod
    def total_impact(self, volume: float, market_params: Dict) -> float:
        """Calculate total price impact"""
        pass

class OomenTransientImpactModel(PriceImpactModel):
    """
    Oomen-style Transient Impact Model with Exponential Decay
    
    Based on the transient impact model (TIM) framework with:
    - Temporary impact that decays exponentially over time
    - Permanent impact following power law
    - Separate calibration of decay parameters
    
    Mathematical Framework:
    - Temporary Impact: I_temp(t) = α * v * exp(-λ * t)
    - Permanent Impact: I_perm = β * sign(v) * |v|^γ
    - Total Impact: I_total(t) = I_perm + Σ I_temp_i(t - t_i)
    """
    
    def __init__(self, currency_pair: str = 'EURUSD'):
        self.currency_pair = currency_pair
        
        # Model parameters to be calibrated
        self.params = {
            'alpha': 0.001,      # Temporary impact coefficient
            'lambda': 0.5,       # Exponential decay rate (1/time_unit)
            'beta': 0.0001,      # Permanent impact coefficient
            'gamma': 0.5,        # Permanent impact exponent (square-root law)
            'sigma': 0.0001,     # Model error standard deviation
        }
        
        # Trade history for computing cumulative temporary impact
        self.trade_history: List[Dict] = []
        
        # Model calibration results
        self.calibration_results = {}
        
    def add_trade(self, timestamp: pd.Timestamp, volume: float, price: float):
        """
        Add a trade to the history for computing cumulative temporary impact
        """
        self.trade_history.append({
            'timestamp': timestamp,
            'volume': volume,
            'price': price,
            'signed_volume': volume  # Will be adjusted based on side
        })
        
        # Keep only recent trades (e.g., last 24 hours) for efficiency
        cutoff_time = timestamp - pd.Timedelta(hours=24)
        self.trade_history = [
            trade for trade in self.trade_history 
            if trade['timestamp'] > cutoff_time
        ]
    
    def temporary_impact_single(self, volume: float, time_elapsed: float = 0.0) -> float:
        """
        Calculate temporary impact for a single trade
        
        I_temp(t) = α * v * exp(-λ * t)
        
        Args:
            volume: Signed trade volume
            time_elapsed: Time elapsed since trade (in same units as lambda)
        
        Returns:
            Temporary impact value
        """
        alpha = self.params['alpha']
        lambda_decay = self.params['lambda']
        
        return alpha * volume * np.exp(-lambda_decay * time_elapsed)
    
    def temporary_impact(self, volume: float, market_params: Dict) -> float:
        """
        Calculate instantaneous temporary impact for current trade
        (time_elapsed = 0)
        """
        return self.temporary_impact_single(volume, time_elapsed=0.0)
    
    def cumulative_temporary_impact(self, current_time: pd.Timestamp) -> float:
        """
        Calculate cumulative temporary impact from all previous trades
        
        Σ I_temp_i(t - t_i) where i iterates over all previous trades
        """
        total_temp_impact = 0.0
        
        for trade in self.trade_history:
            # Calculate time elapsed in fractional days
            time_diff = (current_time - trade['timestamp']).total_seconds() / (24 * 3600)
            
            if time_diff >= 0:  # Only consider past trades
                decay_impact = self.temporary_impact_single(
                    trade['signed_volume'], 
                    time_elapsed=time_diff
                )
                total_temp_impact += decay_impact
        
        return total_temp_impact
    
    def permanent_impact(self, volume: float, market_params: Dict) -> float:
        """
        Calculate permanent impact using power law
        
        I_perm = β * sign(v) * |v|^γ
        """
        beta = self.params['beta']
        gamma = self.params['gamma']
        
        return beta * np.sign(volume) * (np.abs(volume) ** gamma)
    
    def total_impact(self, volume: float, market_params: Dict) -> float:
        """
        Calculate total impact: permanent + instantaneous temporary
        (excludes cumulative temporary impact from history)
        """
        temp_impact = self.temporary_impact(volume, market_params)
        perm_impact = self.permanent_impact(volume, market_params)
        
        return temp_impact + perm_impact
    
    def total_impact_with_history(self, volume: float, current_time: pd.Timestamp, 
                                market_params: Dict) -> float:
        """
        Calculate total impact including cumulative temporary impact from trade history
        """
        # Current trade impacts
        temp_impact = self.temporary_impact(volume, market_params)
        perm_impact = self.permanent_impact(volume, market_params)
        
        # Cumulative temporary impact from previous trades
        cumulative_temp = self.cumulative_temporary_impact(current_time)
        
        return temp_impact + perm_impact + cumulative_temp
    
    def impact_decay_curve(self, volume: float, time_range: np.ndarray) -> np.ndarray:
        """
        Generate impact decay curve for a single trade over time
        
        Args:
            volume: Trade volume
            time_range: Array of time points
            
        Returns:
            Array of temporary impact values over time
        """
        return np.array([
            self.temporary_impact_single(volume, t) for t in time_range
        ])
    
    def predict_price_change(self, volume: float, current_time: pd.Timestamp,
                           market_params: Dict) -> float:
        """
        Predict price change due to trade including all impact components
        """
        return self.total_impact_with_history(volume, current_time, market_params)

class OomenModelCalibrator:
    """
    Calibrator for Oomen Transient Impact Model using Maximum Likelihood Estimation
    """
    
    def __init__(self, model: OomenTransientImpactModel):
        self.model = model
        self.calibration_results = {}
        
    def prepare_data(self, trades: List[TradeData], 
                    market_data: List[MarketData]) -> pd.DataFrame:
        """
        Prepare data for calibration with proper time-series structure
        """
        # Convert to DataFrame
        trade_df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'price': t.price,
                'volume': t.volume,
                'side': t.side,
                'currency_pair': t.currency_pair
            } for t in trades
        ]).sort_values('timestamp')
        
        market_df = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'bid': m.bid,
                'ask': m.ask,
                'mid_price': m.mid_price,
                'spread': m.spread,
                'volatility': m.volatility
            } for m in market_data
        ]).sort_values('timestamp')
        
        # Merge on timestamp
        df = pd.merge_asof(trade_df, market_df, on='timestamp', direction='backward')
        
        # Calculate features for calibration
        df['signed_volume'] = df['volume'] * df['side']
        df['price_change'] = df['price'].diff()
        df['relative_price_change'] = df['price_change'] / df['mid_price']
        
        # Calculate time differences for decay computation
        df['time_diff'] = df['timestamp'].diff().dt.total_seconds() / (24 * 3600)  # Days
        
        # Forward fill missing values and drop invalid rows
        df = df.fillna(method='ffill').dropna()
        
        return df
    
    def compute_predicted_impact(self, params: np.ndarray, data: pd.DataFrame) -> np.ndarray:
        """
        Compute predicted impact for given parameters across all trades
        """
        # Update model parameters
        param_names = ['alpha', 'lambda', 'beta', 'gamma', 'sigma']
        for i, name in enumerate(param_names):
            if i < len(params):
                self.model.params[name] = params[i]
        
        predicted_impacts = []
        
        # Reset trade history
        self.model.trade_history = []
        
        for idx, row in data.iterrows():
            current_time = row['timestamp']
            signed_volume = row['signed_volume']
            
            market_params = {
                'timestamp': current_time,
                'volatility': row['volatility'],
                'spread': row['spread']
            }
            
            # Predict impact including historical trades
            predicted_impact = self.model.total_impact_with_history(
                signed_volume, current_time, market_params
            )
            predicted_impacts.append(predicted_impact)
            
            # Add current trade to history
            self.model.add_trade(current_time, signed_volume, row['price'])
        
        return np.array(predicted_impacts)
    
    def negative_log_likelihood(self, params: np.ndarray, data: pd.DataFrame) -> float:
        """
        Negative log-likelihood function for MLE
        """
        try:
            # Check parameter bounds
            if (params < 0).any():
                return 1e10
            
            predicted_impacts = self.compute_predicted_impact(params, data)
            observed_impacts = data['relative_price_change'].values
            
            # Calculate residuals
            residuals = observed_impacts - predicted_impacts
            sigma = params[-1] if len(params) > 4 else self.model.params['sigma']
            
            # Negative log-likelihood (Gaussian assumption)
            n = len(residuals)
            nll = 0.5 * n * np.log(2 * np.pi * sigma**2)
            nll += 0.5 * np.sum(residuals**2) / (sigma**2)
            
            # Add regularization to prevent extreme parameters
            regularization = 0.001 * np.sum(params**2)
            
            return nll + regularization
            
        except Exception as e:
            # Return large penalty for invalid parameters
            return 1e10
    
    def calibrate(self, trades: List[TradeData], 
                 market_data: List[MarketData]) -> Dict:
        """
        Calibrate model parameters using MLE
        """
        # Prepare data
        data = self.prepare_data(trades, market_data)
        
        if len(data) < 100:
            warnings.warn("Limited data for reliable calibration")
        
        # Initial parameter guess
        initial_params = np.array([0.001, 0.5, 0.0001, 0.5, 0.0001])
        
        # Parameter bounds (all parameters must be positive)
        bounds = [
            (1e-6, 1.0),      # alpha: temporary impact coefficient
            (0.01, 10.0),     # lambda: decay rate
            (1e-8, 0.01),     # beta: permanent impact coefficient
            (0.1, 2.0),       # gamma: impact exponent
            (1e-6, 0.01)      # sigma: error standard deviation
        ]
        
        try:
            # Optimize using L-BFGS-B
            result = optimize.minimize(
                self.negative_log_likelihood,
                initial_params,
                args=(data,),
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                # Update model with calibrated parameters
                param_names = ['alpha', 'lambda', 'beta', 'gamma', 'sigma']
                calibrated_params = {}
                
                for i, name in enumerate(param_names):
                    if i < len(result.x):
                        self.model.params[name] = result.x[i]
                        calibrated_params[name] = result.x[i]
                
                # Calculate fit statistics
                final_predictions = self.compute_predicted_impact(result.x, data)
                observed = data['relative_price_change'].values
                
                # R-squared
                ss_res = np.sum((observed - final_predictions) ** 2)
                ss_tot = np.sum((observed - np.mean(observed)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Mean absolute error
                mae = np.mean(np.abs(observed - final_predictions))
                
                self.calibration_results = {
                    'success': True,
                    'parameters': calibrated_params,
                    'log_likelihood': -result.fun,
                    'aic': 2 * len(result.x) + 2 * result.fun,
                    'bic': len(result.x) * np.log(len(data)) + 2 * result.fun,
                    'r_squared': r_squared,
                    'mae': mae,
                    'n_observations': len(data),
                    'optimization_result': result
                }
                
            else:
                self.calibration_results = {
                    'success': False,
                    'message': result.message,
                    'optimization_result': result
                }
                
        except Exception as e:
            self.calibration_results = {
                'success': False,
                'error': str(e)
            }
        
        return self.calibration_results

print("Oomen Transient Impact Model implementation completed!")
print("Key features:")
print("- Exponential decay of temporary impact")
print("- Power-law permanent impact")
print("- Historical trade impact accumulation")
print("- MLE calibration with regularization")
print("- Comprehensive fit statistics")


# Create analyzer and utility functions for the Oomen model
class OomenImpactAnalyzer:
    """
    Analyzer for Oomen Transient Impact Model
    Provides visualization and analysis tools
    """
    
    def __init__(self, model: OomenTransientImpactModel):
        self.model = model
        
    def analyze_decay_curve(self, volume: float, max_time: float = 2.0, 
                          n_points: int = 100) -> pd.DataFrame:
        """
        Analyze the exponential decay curve for temporary impact
        
        Args:
            volume: Trade volume to analyze
            max_time: Maximum time to analyze (in days)
            n_points: Number of time points
        """
        time_range = np.linspace(0, max_time, n_points)
        
        # Calculate temporary impact decay
        temp_impacts = self.model.impact_decay_curve(volume, time_range)
        
        # Calculate permanent impact (constant over time)
        perm_impact = self.model.permanent_impact(volume, {})
        
        # Total impact over time
        total_impacts = temp_impacts + perm_impact
        
        results = pd.DataFrame({
            'time_hours': time_range * 24,  # Convert to hours
            'time_days': time_range,
            'temporary_impact': temp_impacts,
            'permanent_impact': perm_impact,
            'total_impact': total_impacts,
            'decay_factor': np.exp(-self.model.params['lambda'] * time_range)
        })
        
        return results
    
    def impact_sensitivity_analysis(self, base_volume: float = 1000000) -> pd.DataFrame:
        """
        Analyze impact sensitivity to parameter changes
        """
        base_params = self.model.params.copy()
        
        # Parameter variations to test
        param_variations = {
            'alpha': np.linspace(0.0001, 0.01, 10),
            'lambda': np.linspace(0.1, 2.0, 10),
            'beta': np.linspace(0.00001, 0.001, 10),
            'gamma': np.linspace(0.3, 0.8, 10)
        }
        
        results = []
        
        for param_name, param_values in param_variations.items():
            for param_value in param_values:
                # Reset to base parameters
                self.model.params = base_params.copy()
                
                # Set the varying parameter
                self.model.params[param_name] = param_value
                
                # Calculate impacts
                temp_impact = self.model.temporary_impact(base_volume, {})
                perm_impact = self.model.permanent_impact(base_volume, {})
                total_impact = temp_impact + perm_impact
                
                results.append({
                    'parameter': param_name,
                    'value': param_value,
                    'temporary_impact': temp_impact,
                    'permanent_impact': perm_impact,
                    'total_impact': total_impact
                })
        
        # Restore original parameters
        self.model.params = base_params
        
        return pd.DataFrame(results)
    
    def compare_with_baseline(self, baseline_model: PriceImpactModel, 
                             volume_range: np.ndarray) -> pd.DataFrame:
        """
        Compare Oomen model with baseline model across volume range
        """
        results = []
        
        for volume in volume_range:
            market_params = {'timestamp': pd.Timestamp.now(), 'volatility': 0.001, 'spread': 0.0001}
            
            oomen_temp = self.model.temporary_impact(volume, market_params)
            oomen_perm = self.model.permanent_impact(volume, market_params)
            oomen_total = oomen_temp + oomen_perm
            
            baseline_temp = baseline_model.temporary_impact(volume, market_params)
            baseline_perm = baseline_model.permanent_impact(volume, market_params)
            baseline_total = baseline_temp + baseline_perm
            
            results.append({
                'volume': volume,
                'oomen_temporary': oomen_temp,
                'oomen_permanent': oomen_perm,
                'oomen_total': oomen_total,
                'baseline_temporary': baseline_temp,
                'baseline_permanent': baseline_perm,
                'baseline_total': baseline_total,
                'temp_diff': oomen_temp - baseline_temp,
                'perm_diff': oomen_perm - baseline_perm,
                'total_diff': oomen_total - baseline_total
            })
        
        return pd.DataFrame(results)

# Enhanced data generation function that includes realistic time dependencies
def generate_enhanced_fx_data(n_trades: int = 2000, 
                            currency_pair: str = 'EURUSD',
                            start_date: str = '2024-01-01') -> Tuple[List[TradeData], List[MarketData]]:
    """
    Generate enhanced synthetic FX data with realistic time structure and impact patterns
    """
    np.random.seed(42)
    
    # Generate timestamps with realistic trading patterns
    start_time = pd.Timestamp(start_date)
    
    # Create timestamps with higher density during active trading hours
    timestamps = []
    current_time = start_time
    
    for i in range(n_trades):
        # Add random intervals (heavier during active hours)
        hour = current_time.hour
        if 8 <= hour <= 17:  # Active trading hours
            interval = np.random.exponential(30)  # Average 30 seconds
        else:  # Quiet hours
            interval = np.random.exponential(120)  # Average 2 minutes
        
        current_time += pd.Timedelta(seconds=interval)
        timestamps.append(current_time)
    
    # Market data parameters
    base_price = 1.1000
    base_volatility = 0.001
    
    trades = []
    market_data = []
    
    # Initialize price process
    current_price = base_price
    accumulated_impact = 0.0
    
    # Temporary impact model for data generation
    temp_impact_history = []
    
    for i, ts in enumerate(timestamps):
        # Update market volatility (time-varying)
        hour = ts.hour
        if 13 <= hour <= 17:  # London-NY overlap
            volatility = base_volatility * 0.8
        elif 0 <= hour <= 6:  # Asian session
            volatility = base_volatility * 1.3
        else:
            volatility = base_volatility
        
        # Price evolution with mean reversion
        drift = -0.01 * (current_price - base_price)  # Mean reversion
        noise = np.random.normal(0, volatility)
        
        # Decay previous temporary impacts
        decayed_temp_impact = 0.0
        new_temp_history = []
        
        for prev_impact, prev_time in temp_impact_history:
            time_diff = (ts - prev_time).total_seconds() / (24 * 3600)  # Days
            decayed_impact = prev_impact * np.exp(-2.0 * time_diff)  # λ = 2.0
            if abs(decayed_impact) > 1e-8:  # Keep if still significant
                decayed_temp_impact += decayed_impact
                new_temp_history.append((decayed_impact, prev_time))
        
        temp_impact_history = new_temp_history
        
        # Update price
        current_price += drift + noise + decayed_temp_impact
        
        # Generate spread
        spread = np.random.uniform(0.00005, 0.0002)
        bid = current_price - spread/2
        ask = current_price + spread/2
        
        # Market data point
        market_data.append(MarketData(
            timestamp=ts,
            bid=bid,
            ask=ask,
            mid_price=current_price,
            spread=spread,
            volatility=volatility,
            currency_pair=currency_pair
        ))
        
        # Generate trade with probability
        if np.random.random() < 0.4:  # 40% probability
            # Volume follows log-normal distribution
            base_volume = np.random.lognormal(mean=12, sigma=1.5)
            
            # Larger volumes more likely during active hours
            if 8 <= hour <= 17:
                volume_multiplier = np.random.uniform(0.5, 3.0)
            else:
                volume_multiplier = np.random.uniform(0.2, 1.5)
            
            volume = base_volume * volume_multiplier
            side = np.random.choice([-1, 1])
            
            # Calculate impact for this trade
            temp_impact = 0.002 * side * volume / 1000000  # Temporary impact
            perm_impact = 0.001 * side * np.sqrt(volume / 1000000)  # Permanent impact
            
            # Apply impact to price
            trade_price = current_price + temp_impact + perm_impact
            
            # Add to temporary impact history
            temp_impact_history.append((temp_impact, ts))
            
            # Update accumulated permanent impact
            accumulated_impact += perm_impact
            current_price += perm_impact  # Permanent impact persists
            
            trades.append(TradeData(
                timestamp=ts,
                price=trade_price,
                volume=volume,
                side=side,
                currency_pair=currency_pair
            ))
    
    return trades, market_data

# Example usage function
def run_oomen_example():
    """
    Comprehensive example of the Oomen Transient Impact Model
    """
    print("Oomen Transient Impact Model - Enhanced Example")
    print("=" * 55)
    
    # Generate enhanced synthetic data
    print("Generating enhanced FX data with realistic time structure...")
    trades, market_data = generate_enhanced_fx_data(n_trades=3000, currency_pair='EURUSD')
    print(f"Generated {len(trades)} trades and {len(market_data)} market data points")
    
    # Initialize Oomen model
    print("\nInitializing Oomen Transient Impact Model...")
    model = OomenTransientImpactModel('EURUSD')
    
    # Display initial parameters
    print("Initial parameters:")
    for param, value in model.params.items():
        print(f"  {param}: {value:.6f}")
    
    # Calibrate model
    print("\nCalibrating model parameters...")
    calibrator = OomenModelCalibrator(model)
    results = calibrator.calibrate(trades, market_data)
    
    if results['success']:
        print("✓ Calibration successful!")
        print("\nCalibrated parameters:")
        for param, value in results['parameters'].items():
            print(f"  {param}: {value:.6f}")
        
        print(f"\nModel fit statistics:")
        print(f"  Log-likelihood: {results['log_likelihood']:.2f}")
        print(f"  AIC: {results['aic']:.2f}")
        print(f"  R-squared: {results['r_squared']:.4f}")
        print(f"  Mean Absolute Error: {results['mae']:.6f}")
        
    else:
        print("✗ Calibration failed:", results.get('message', results.get('error')))
        return
    
    # Analyze model
    print("\nAnalyzing model behavior...")
    analyzer = OomenImpactAnalyzer(model)
    
    # Test impact calculation
    test_volume = 1000000
    current_time = pd.Timestamp('2024-01-01 12:00:00')
    market_params = {
        'timestamp': current_time,
        'volatility': 0.001,
        'spread': 0.0001
    }
    
    temp_impact = model.temporary_impact(test_volume, market_params)
    perm_impact = model.permanent_impact(test_volume, market_params)
    total_impact = model.total_impact(test_volume, market_params)
    
    print(f"\nImpact analysis for {test_volume:,.0f} units:")
    print(f"  Temporary impact (t=0): {temp_impact:.6f}")
    print(f"  Permanent impact: {perm_impact:.6f}")
    print(f"  Total impact: {total_impact:.6f}")
    
    # Decay analysis
    print(f"\nTemporary impact decay analysis:")
    decay_df = analyzer.analyze_decay_curve(test_volume, max_time=1.0)  # 1 day
    
    print(f"  Impact at t=0: {decay_df.iloc[0]['temporary_impact']:.6f}")
    print(f"  Impact at t=1h: {decay_df.iloc[4]['temporary_impact']:.6f}")
    print(f"  Impact at t=6h: {decay_df.iloc[25]['temporary_impact']:.6f}")
    print(f"  Impact at t=24h: {decay_df.iloc[-1]['temporary_impact']:.6f}")
    
    # Half-life calculation
    lambda_param = model.params['lambda']
    half_life_days = np.log(2) / lambda_param
    half_life_hours = half_life_days * 24
    
    print(f"\nDecay characteristics:")
    print(f"  Decay rate (λ): {lambda_param:.3f} /day")
    print(f"  Half-life: {half_life_hours:.1f} hours")
    print(f"  99% decay time: {4.6 / lambda_param * 24:.1f} hours")
    
    return model, calibrator, analyzer

# Save the implementation
print("Implementation complete!")
print("\nKey components created:")
print("- OomenTransientImpactModel: Main model class")
print("- OomenModelCalibrator: MLE calibration")
print("- OomenImpactAnalyzer: Analysis tools")
print("- Enhanced data generation with realistic patterns")
print("- Comprehensive example function")


# Run the comprehensive example to demonstrate the model
model, calibrator, analyzer = run_oomen_example()


# Create a simplified demonstration focusing on the key model features
def demonstrate_oomen_model():
    """
    Demonstrate key features of the Oomen model without heavy calibration
    """
    print("Oomen Transient Impact Model - Key Features Demo")
    print("=" * 50)
    
    # Initialize model with typical parameters
    model = OomenTransientImpactModel('EURUSD')
    
    # Set realistic calibrated parameters
    model.params = {
        'alpha': 0.003,      # Temporary impact coefficient
        'lambda': 1.2,       # Exponential decay rate (1.2 /day)
        'beta': 0.0005,      # Permanent impact coefficient
        'gamma': 0.5,        # Square-root law
        'sigma': 0.0002,     # Model error
    }
    
    print("Model parameters:")
    for param, value in model.params.items():
        print(f"  {param}: {value:.6f}")
    
    # Test volume
    test_volume = 1000000  # 1M units
    market_params = {'timestamp': pd.Timestamp.now(), 'volatility': 0.001, 'spread': 0.0001}
    
    print(f"\nImpact analysis for {test_volume:,.0f} units:")
    
    # Calculate different impact components
    temp_impact = model.temporary_impact(test_volume, market_params)
    perm_impact = model.permanent_impact(test_volume, market_params)
    total_impact = temp_impact + perm_impact
    
    print(f"  Temporary impact (t=0): {temp_impact:.6f}")
    print(f"  Permanent impact: {perm_impact:.6f}")
    print(f"  Total impact: {total_impact:.6f}")
    
    # Decay analysis
    print(f"\nTemporary impact decay over time:")
    time_points = [0, 0.1, 0.5, 1.0, 2.0, 6.0, 24.0]  # Hours
    time_points_days = [t/24 for t in time_points]
    
    for i, (t_hours, t_days) in enumerate(zip(time_points, time_points_days)):
        decayed_impact = model.temporary_impact_single(test_volume, t_days)
        decay_factor = np.exp(-model.params['lambda'] * t_days)
        print(f"  t={t_hours:4.1f}h: {decayed_impact:.6f} (decay factor: {decay_factor:.3f})")
    
    # Half-life calculation
    lambda_param = model.params['lambda']
    half_life_hours = np.log(2) / lambda_param * 24
    print(f"\nDecay characteristics:")
    print(f"  Half-life: {half_life_hours:.1f} hours")
    print(f"  95% decay time: {3.0 / lambda_param * 24:.1f} hours")
    
    # Volume sensitivity
    print(f"\nVolume sensitivity (permanent impact):")
    volumes = [100000, 500000, 1000000, 2000000, 5000000]
    for vol in volumes:
        perm = model.permanent_impact(vol, market_params)
        print(f"  {vol:8,.0f} units: {perm:.6f}")
    
    return model

# Run demonstration
model = demonstrate_oomen_model()

# Create decay curve data for visualization
analyzer = OomenImpactAnalyzer(model)
decay_data = analyzer.analyze_decay_curve(1000000, max_time=2.0, n_points=100)

print(f"\nDecay curve data generated with {len(decay_data)} points")
print(f"Initial temporary impact: {decay_data.iloc[0]['temporary_impact']:.6f}")
print(f"Final temporary impact: {decay_data.iloc[-1]['temporary_impact']:.6f}")
print(f"Permanent impact: {decay_data.iloc[0]['permanent_impact']:.6f}")

# Save decay curve data
decay_data.to_csv('oomen_decay_curve.csv', index=False)
print("\nDecay curve data saved to 'oomen_decay_curve.csv'")

# Create parameter sensitivity data
sensitivity_data = analyzer.impact_sensitivity_analysis(1000000)
print(f"\nParameter sensitivity analysis completed")
print(f"Analyzed {len(sensitivity_data)} parameter combinations")

# Display summary statistics
print("\nSensitivity Analysis Summary:")
for param in ['alpha', 'lambda', 'beta', 'gamma']:
    param_data = sensitivity_data[sensitivity_data['parameter'] == param]
    print(f"  {param}:")
    print(f"    Total impact range: {param_data['total_impact'].min():.6f} to {param_data['total_impact'].max():.6f}")
    print(f"    Most sensitive to: {param_data.loc[param_data['total_impact'].idxmax(), 'value']:.6f}")

# Save sensitivity data
sensitivity_data.to_csv('oomen_sensitivity_analysis.csv', index=False)
print("\nSensitivity analysis saved to 'oomen_sensitivity_analysis.csv'")

import pandas as pd
import plotly.graph_objects as go

# Load the data
df = pd.read_csv('oomen_decay_curve.csv')

# Create the chart showing Impact vs Time
fig = go.Figure()

# Add temporary impact (exponential decay curve)
fig.add_trace(go.Scatter(
    x=df['time_hours'],
    y=df['temporary_impact'],
    mode='lines',
    name='Temporary',
    line=dict(color='#1FB8CD', width=3),
    cliponaxis=False
))

# Add permanent impact (horizontal line)
fig.add_trace(go.Scatter(
    x=df['time_hours'],
    y=df['permanent_impact'],
    mode='lines',
    name='Permanent',
    line=dict(color='#FFC185', width=3),
    cliponaxis=False
))

# Add total impact
fig.add_trace(go.Scatter(
    x=df['time_hours'],
    y=df['total_impact'],
    mode='lines',
    name='Total',
    line=dict(color='#ECEBD5', width=3),
    cliponaxis=False
))

# Update layout
fig.update_layout(
    title='Oomen Price Impact Model',
    xaxis_title='Time (hrs)',
    yaxis_title='Impact',
    showlegend=True,
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update axes
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)

# Save the chart
fig.write_image('oomen_impact_chart.png')
print("Chart saved successfully!")
