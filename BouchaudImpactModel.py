import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

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
        """Calculate permanent price impact"""
        pass
    
    @abstractmethod
    def total_impact(self, volume: float, market_params: Dict) -> float:
        """Calculate total price impact"""
        pass

class BouchaudImpactModel(PriceImpactModel):
    """
    Bouchaud et al. price impact model implementation
    
    Based on the square-root law: Impact ∝ √Q
    Incorporates both temporary and permanent components
    """
    
    def __init__(self):
        # Model parameters to be calibrated
        self.params = {
            'lambda_temp': 0.1,      # Temporary impact coefficient
            'lambda_perm': 0.05,     # Permanent impact coefficient  
            'gamma': 0.5,            # Square-root exponent
            'eta': 0.1,              # Linear temporary impact
            'sigma': 0.001,          # Price volatility
            'participation_rate': 0.1 # Trading intensity
        }
        
    def temporary_impact(self, volume: float, market_params: Dict) -> float:
        """
        Temporary impact following Almgren-Chriss formulation
        I_temp = η * v + λ_temp * |v|^γ
        """
        eta = self.params['eta']
        lambda_temp = self.params['lambda_temp']
        gamma = self.params['gamma']
        
        # Linear component + non-linear component
        linear_impact = eta * volume
        nonlinear_impact = lambda_temp * np.sign(volume) * (np.abs(volume) ** gamma)
        
        return linear_impact + nonlinear_impact
    
    def permanent_impact(self, volume: float, market_params: Dict) -> float:
        """
        Permanent impact based on square-root law
        I_perm = λ_perm * sign(v) * √|v|
        """
        lambda_perm = self.params['lambda_perm']
        return lambda_perm * np.sign(volume) * np.sqrt(np.abs(volume))
    
    def total_impact(self, volume: float, market_params: Dict) -> float:
        """Total impact combining temporary and permanent components"""
        temp_impact = self.temporary_impact(volume, market_params)
        perm_impact = self.permanent_impact(volume, market_params)
        return temp_impact + perm_impact
    
    def metaorder_impact(self, total_volume: float, num_slices: int, 
                        market_params: Dict) -> Tuple[float, List[float]]:
        """
        Calculate impact for a metaorder executed in slices
        Implements the square-root law: impact depends on total volume, 
        not execution schedule
        """
        slice_volume = total_volume / num_slices
        slice_impacts = []
        
        # Temporary impact accumulates with each slice
        cumulative_temp_impact = 0
        
        for i in range(num_slices):
            # Temporary impact for this slice
            temp_impact = self.temporary_impact(slice_volume, market_params)
            slice_impacts.append(temp_impact)
            cumulative_temp_impact += temp_impact
        
        # Permanent impact depends on total volume (square-root law)
        total_perm_impact = self.permanent_impact(total_volume, market_params)
        
        # Total metaorder impact
        total_impact = cumulative_temp_impact / num_slices + total_perm_impact
        
        return total_impact, slice_impacts

class FXMarketImpactModel(BouchaudImpactModel):
    """
    Specialized price impact model for FX spot markets
    
    Incorporates FX-specific features:
    - Currency pair characteristics
    - Time-of-day effects
    - Cross-currency correlations
    """
    
    def __init__(self, currency_pair: str):
        super().__init__()
        self.currency_pair = currency_pair
        self.fx_params = self._initialize_fx_params()
        
    def _initialize_fx_params(self) -> Dict:
        """Initialize FX-specific parameters"""
        # Major pairs have different liquidity characteristics
        major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']
        
        if self.currency_pair in major_pairs:
            return {
                'liquidity_factor': 1.0,
                'spread_multiplier': 1.0,
                'volatility_regime': 'normal'
            }
        else:
            return {
                'liquidity_factor': 1.5,  # Higher impact for minor pairs
                'spread_multiplier': 1.2,
                'volatility_regime': 'elevated'
            }
    
    def adjust_for_time_of_day(self, timestamp: pd.Timestamp) -> float:
        """
        Adjust impact for FX market sessions
        Higher impact during low-liquidity periods
        """
        hour = timestamp.hour
        
        # London-NY overlap (high liquidity)
        if 13 <= hour <= 17:
            return 0.8
        # Asian session (lower liquidity for major pairs)
        elif 0 <= hour <= 6:
            return 1.3
        # London session
        elif 7 <= hour <= 12:
            return 0.9
        # NY session
        else:
            return 1.0
    
    def temporary_impact(self, volume: float, market_params: Dict) -> float:
        """FX-adjusted temporary impact"""
        base_impact = super().temporary_impact(volume, market_params)
        
        # Adjust for FX-specific factors
        liquidity_adj = self.fx_params['liquidity_factor']
        time_adj = self.adjust_for_time_of_day(market_params.get('timestamp', pd.Timestamp.now()))
        
        return base_impact * liquidity_adj * time_adj

class ModelCalibrator:
    """
    Calibrates price impact model parameters using empirical data
    Implements Maximum Likelihood Estimation (MLE) approach
    """
    
    def __init__(self, model: PriceImpactModel):
        self.model = model
        self.calibration_results = {}
        
    def prepare_data(self, trades: List[TradeData], 
                    market_data: List[MarketData]) -> pd.DataFrame:
        """
        Prepare data for calibration
        Combines trade and market data into analysis-ready format
        """
        trade_df = pd.DataFrame([
            {
                'timestamp': t.timestamp,
                'price': t.price,
                'volume': t.volume,
                'side': t.side,
                'currency_pair': t.currency_pair
            } for t in trades
        ])
        
        market_df = pd.DataFrame([
            {
                'timestamp': m.timestamp,
                'bid': m.bid,
                'ask': m.ask,
                'mid_price': m.mid_price,
                'spread': m.spread,
                'volatility': m.volatility
            } for m in market_data
        ])
        
        # Merge on timestamp
        df = pd.merge_asof(trade_df.sort_values('timestamp'),
                          market_df.sort_values('timestamp'),
                          on='timestamp',
                          direction='backward')
        
        # Calculate observed impact
        df['price_change'] = df['price'].diff()
        df['signed_volume'] = df['volume'] * df['side']
        df['observed_impact'] = df['price_change'] / df['mid_price']  # Relative impact
        
        return df.dropna()
    
    def negative_log_likelihood(self, params: np.ndarray, data: pd.DataFrame) -> float:
        """
        Negative log-likelihood function for MLE
        Assumes Gaussian errors around predicted impact
        """
        # Update model parameters
        param_names = ['lambda_temp', 'lambda_perm', 'gamma', 'eta', 'sigma']
        for i, name in enumerate(param_names):
            if i < len(params):
                self.model.params[name] = params[i]
        
        # Calculate predicted impacts
        predicted_impacts = []
        for _, row in data.iterrows():
            market_params = {
                'timestamp': row['timestamp'],
                'volatility': row['volatility'],
                'spread': row['spread']
            }
            predicted_impact = self.model.total_impact(row['signed_volume'], market_params)
            predicted_impacts.append(predicted_impact)
        
        predicted_impacts = np.array(predicted_impacts)
        observed_impacts = data['observed_impact'].values
        
        # Calculate residuals
        residuals = observed_impacts - predicted_impacts
        sigma = self.model.params['sigma']
        
        # Negative log-likelihood (assuming Gaussian errors)
        nll = 0.5 * len(residuals) * np.log(2 * np.pi * sigma**2)
        nll += 0.5 * np.sum(residuals**2) / sigma**2
        
        return nll
    
    def calibrate(self, trades: List[TradeData], 
                 market_data: List[MarketData]) -> Dict:
        """
        Calibrate model parameters using MLE
        """
        # Prepare data
        data = self.prepare_data(trades, market_data)
        
        if len(data) < 50:
            warnings.warn("Insufficient data for reliable calibration")
        
        # Initial parameter guess
        initial_params = np.array([0.1, 0.05, 0.5, 0.1, 0.001])
        
        # Parameter bounds
        bounds = [
            (0.001, 1.0),  # lambda_temp
            (0.001, 0.5),  # lambda_perm  
            (0.1, 1.0),    # gamma
            (0.001, 0.5),  # eta
            (0.0001, 0.01) # sigma
        ]
        
        # Optimize
        try:
            result = optimize.minimize(
                self.negative_log_likelihood,
                initial_params,
                args=(data,),
                bounds=bounds,
                method='L-BFGS-B'
            )
            
            if result.success:
                # Update model with calibrated parameters
                param_names = ['lambda_temp', 'lambda_perm', 'gamma', 'eta', 'sigma']
                for i, name in enumerate(param_names):
                    if i < len(result.x):
                        self.model.params[name] = result.x[i]
                
                # Calculate fit statistics
                self.calibration_results = {
                    'success': True,
                    'parameters': dict(zip(param_names, result.x)),
                    'log_likelihood': -result.fun,
                    'aic': 2 * len(result.x) + 2 * result.fun,
                    'bic': len(result.x) * np.log(len(data)) + 2 * result.fun,
                    'n_observations': len(data)
                }
                
            else:
                self.calibration_results = {
                    'success': False,
                    'message': result.message
                }
                
        except Exception as e:
            self.calibration_results = {
                'success': False,
                'error': str(e)
            }
        
        return self.calibration_results

class ImpactAnalyzer:
    """
    Analyzes and visualizes price impact results
    """
    
    def __init__(self, model: PriceImpactModel):
        self.model = model
        
    def analyze_impact_curve(self, volume_range: np.ndarray, 
                           market_params: Dict) -> pd.DataFrame:
        """
        Generate impact curve over volume range
        """
        results = []
        
        for volume in volume_range:
            temp_impact = self.model.temporary_impact(volume, market_params)
            perm_impact = self.model.permanent_impact(volume, market_params)
            total_impact = temp_impact + perm_impact
            
            results.append({
                'volume': volume,
                'temporary_impact': temp_impact,
                'permanent_impact': perm_impact,
                'total_impact': total_impact
            })
        
        return pd.DataFrame(results)
    
    def plot_impact_curves(self, volume_range: np.ndarray, 
                          market_params: Dict) -> plt.Figure:
        """
        Plot impact curves
        """
        df = self.analyze_impact_curve(volume_range, market_params)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Impact vs Volume
        ax1.plot(df['volume'], df['temporary_impact'], 
                label='Temporary Impact', color='blue')
        ax1.plot(df['volume'], df['permanent_impact'], 
                label='Permanent Impact', color='red')
        ax1.plot(df['volume'], df['total_impact'], 
                label='Total Impact', color='green', linewidth=2)
        
        ax1.set_xlabel('Volume')
        ax1.set_ylabel('Price Impact')
        ax1.set_title('Price Impact vs Volume')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Log-log plot to verify square-root law
        ax2.loglog(df['volume'], df['permanent_impact'], 
                  'r-', label='Permanent Impact')
        ax2.loglog(df['volume'], df['volume']**0.5 * 0.01, 
                  'k--', label='√V (reference)', alpha=0.7)
        
        ax2.set_xlabel('Volume (log scale)')
        ax2.set_ylabel('Impact (log scale)')
        ax2.set_title('Square-Root Law Verification')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Utility functions for data generation and testing

def generate_synthetic_fx_data(n_trades: int = 1000, 
                              currency_pair: str = 'EURUSD') -> Tuple[List[TradeData], List[MarketData]]:
    """
    Generate synthetic FX trade and market data for testing
    """
    np.random.seed(42)
    
    # Generate timestamps
    start_time = pd.Timestamp('2024-01-01 00:00:00')
    timestamps = pd.date_range(start_time, periods=n_trades, freq='1s')
    
    # Generate synthetic market data
    base_price = 1.1000  # EUR/USD
    volatility = 0.001
    
    market_data = []
    trades = []
    
    current_price = base_price
    
    for i, ts in enumerate(timestamps):
        # Market data evolution
        price_change = np.random.normal(0, volatility)
        current_price += price_change
        
        spread = np.random.uniform(0.00005, 0.0002)  # 0.5-2 pips
        bid = current_price - spread/2
        ask = current_price + spread/2
        
        market_data.append(MarketData(
            timestamp=ts,
            bid=bid,
            ask=ask,
            mid_price=current_price,
            spread=spread,
            volatility=volatility,
            currency_pair=currency_pair
        ))
        
        # Generate trade (not every timestamp)
        if np.random.random() < 0.3:  # 30% probability of trade
            volume = np.random.lognormal(mean=10, sigma=1)  # Lognormal volume
            side = np.random.choice([-1, 1])  # Random buy/sell
            
            # Trade price includes impact
            trade_price = current_price + side * spread/2
            
            trades.append(TradeData(
                timestamp=ts,
                price=trade_price,
                volume=volume,
                side=side,
                currency_pair=currency_pair
            ))
    
    return trades, market_data

def run_example():
    """
    Example usage of the price impact modeling framework
    """
    print("High Frequency Price Impact Model - FX Example")
    print("=" * 50)
    
    # Generate synthetic data
    print("Generating synthetic FX data...")
    trades, market_data = generate_synthetic_fx_data(n_trades=2000, currency_pair='EURUSD')
    print(f"Generated {len(trades)} trades and {len(market_data)} market data points")
    
    # Initialize FX-specific model
    print("\nInitializing FX price impact model...")
    model = FXMarketImpactModel('EURUSD')
    
    # Calibrate model
    print("Calibrating model parameters...")
    calibrator = ModelCalibrator(model)
    results = calibrator.calibrate(trades, market_data)
    
    if results['success']:
        print("Calibration successful!")
        print("Calibrated parameters:")
        for param, value in results['parameters'].items():
            print(f"  {param}: {value:.6f}")
        print(f"Log-likelihood: {results['log_likelihood']:.2f}")
        print(f"AIC: {results['aic']:.2f}")
    else:
        print("Calibration failed:", results.get('message', results.get('error')))
    
    # Analyze impact
    print("\nAnalyzing price impact...")
    analyzer = ImpactAnalyzer(model)
    
    # Test impact calculation
    test_volume = 1000000  # 1M units
    market_params = {
        'timestamp': pd.Timestamp('2024-01-01 12:00:00'),
        'volatility': 0.001,
        'spread': 0.0001
    }
    
    temp_impact = model.temporary_impact(test_volume, market_params)
    perm_impact = model.permanent_impact(test_volume, market_params)
    total_impact = model.total_impact(test_volume, market_params)
    
    print(f"\nImpact analysis for {test_volume:,.0f} units:")
    print(f"  Temporary impact: {temp_impact:.6f}")
    print(f"  Permanent impact: {perm_impact:.6f}")  
    print(f"  Total impact: {total_impact:.6f}")
    
    # Metaorder analysis
    print(f"\nMetaorder analysis (sliced execution):")
    total_impact_meta, slice_impacts = model.metaorder_impact(
        total_volume=test_volume, 
        num_slices=10, 
        market_params=market_params
    )
    print(f"  Total metaorder impact: {total_impact_meta:.6f}")
    print(f"  Average slice impact: {np.mean(slice_impacts):.6f}")
    
    print("\n" + "=" * 50)
    print("Analysis complete!")

if __name__ == "__main__":
    run_example()
