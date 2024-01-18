pub use ndarray::array;
pub use ndarray::{arr1, s, ArrayBase, Ix1, OwnedRepr, ViewRepr};

/// Error type for equity indicators
#[non_exhaustive]
#[derive(Debug)]
pub enum IndicatorError {
    /// Indicates that not enough data points were provided to an indicator function to satisfy the
    /// given window
    NotEnoughData(String),
}

impl std::fmt::Display for IndicatorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for IndicatorError {}

/// Calculates the Relative Strength Index (RSI) for a given price array and window size.
///
/// # Arguments
///
/// * `prices` - An array of price data.
/// * `window` - The size of the window for calculating RSI.
///
/// # Returns
///
/// A Result containing an array of RSI values or an `IndicatorError` if there is not enough data.
///
/// # Errors
///
/// Returns an `IndicatorError::NotEnoughData` if the length of `prices` is less than or equal to
/// `window`.
pub fn calculate_rsi(
    prices: ArrayBase<ViewRepr<&f64>, Ix1>,
    window: usize,
) -> Result<ArrayBase<OwnedRepr<f64>, Ix1>, IndicatorError> {
    // Check if prices array has enough elements
    if prices.len() <= window {
        return Err(IndicatorError::NotEnoughData(
            "Not enough data points to caluclate RSI".to_string(),
        ));
    }

    // Calculate price changes
    let price_changes = prices.slice(s![1..]).to_owned() - prices.slice(s![..-1]);

    // Separate gains and losses
    let gains = price_changes.mapv(|x| if x > 0.0 { x } else { 0.0 });
    let losses = price_changes.mapv(|x| if x < 0.0 { -x } else { 0.0 });

    // Calculate average gains and losses over the window
    let mut avg_gain = gains.slice(s![..window]).sum() / window as f64;
    let mut avg_loss = losses.slice(s![..window]).sum() / window as f64;

    // Calculate RSI for each element in the specified window to the end
    let mut rsi_values = Vec::with_capacity(prices.len() - window);
    for i in window..prices.len() {
        let current_gain = gains[i - 1];
        let current_loss = losses[i - 1];

        // Calculate average gains and losses using the previous averages
        avg_gain = ((avg_gain * (window - 1) as f64) + current_gain) / window as f64;
        avg_loss = ((avg_loss * (window - 1) as f64) + current_loss) / window as f64;

        // Calculate RS and RSI for the current element
        let rs = if avg_loss > 0.0 {
            avg_gain / avg_loss
        } else {
            f64::INFINITY
        };
        let rsi = 100.0 - (100.0 / (1.0 + rs));

        rsi_values.push(rsi);
    }

    // Convert the Vec<f64> to an ArrayBase
    Ok(arr1(&rsi_values))
}

/// Calculates the Exponential Moving Average (EMA) for a given price array and window size.
///
/// # Arguments
///
/// * `prices` - An array of price data.
/// * `window` - The size of the window for calculating EMA.
///
/// # Returns
///
/// A Result containing an array of EMA values or an `IndicatorError` if there is not enough data.
///
/// # Errors
///
/// Returns an `IndicatorError::NotEnoughData` if the length of `prices` is less than `window`.
pub fn calculate_ema(
    prices: ArrayBase<ViewRepr<&f64>, Ix1>,
    window: usize,
) -> Result<ArrayBase<OwnedRepr<f64>, Ix1>, IndicatorError> {
    if prices.len() < window {
        return Err(IndicatorError::NotEnoughData(
            "`prices` must have at least `window` items".to_string(),
        ));
    }

    let smoothing = 2.0 / (window as f64 + 1.0);

    let sma = prices.slice(s![..window]).sum() / window as f64;
    let mut ema_values = Vec::with_capacity(prices.len() - window);
    ema_values.push(sma);

    for i in window..prices.len() {
        let current_price = prices[i];
        let prev_ema = ema_values[i - window];

        let ema = (current_price - prev_ema) * smoothing + prev_ema;
        ema_values.push(ema);
    }

    Ok(arr1(&ema_values))
}

/// Calculates the Moving Average Convergence Divergence (MACD) for a given price array and parameters.
///
/// # Arguments
///
/// * `prices` - An array of price data.
/// * `short_window` - The size of the short-term EMA window.
/// * `long_window` - The size of the long-term EMA window.
/// * `signal_window` - The size of the signal line window.
///
/// # Returns
///
/// A Result containing a tuple of MACD line, signal line, and histogram or an `IndicatorError` if there is not enough data.
///
/// # Errors
///
/// Returns an `IndicatorError::NotEnoughData` if the length of `prices` is insufficient to
/// calculate any of the moving averages for the `short_window`, `long_window`, or the `signal_window`.
pub fn calculate_macd(
    prices: ArrayBase<ViewRepr<&f64>, Ix1>,
    short_window: usize,
    long_window: usize,
    signal_window: usize,
) -> Result<
    (
        ArrayBase<OwnedRepr<f64>, Ix1>,
        ArrayBase<OwnedRepr<f64>, Ix1>,
        ArrayBase<OwnedRepr<f64>, Ix1>,
    ),
    IndicatorError,
> {
    let mut ema_short = calculate_ema(prices, short_window)?;
    let ema_long = calculate_ema(prices, long_window)?;
    ema_short = ema_short.slice(s![long_window - short_window..]).to_owned();

    let mut macd_line = ema_short - ema_long;
    let signal_line = calculate_ema(macd_line.slice(s![..]), signal_window)?;
    macd_line = macd_line
        .slice(s![macd_line.len() - signal_line.len()..])
        .to_owned();

    let histogram = macd_line.clone() - signal_line.clone();
    Ok((macd_line, signal_line, histogram))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_calculate_rsi() {
        // Test case with enough data
        let prices = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let window = 3;
        let result = calculate_rsi(prices.view(), window).unwrap();
        assert_eq!(result, array![100.0, 100.0]);

        // Test case with not enough data
        let prices = array![1.0, 2.0];
        let window = 3;
        let result = calculate_rsi(prices.view(), window);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::NotEnoughData(_)) => (),
            _ => {
                panic!("Expected `IndicatorError::NotEnoughData`, found different `IndicatorError`")
            }
        }
    }

    #[test]
    fn test_calculate_ema() {
        // Test case with enough data
        let prices = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let window = 3;
        let result = calculate_ema(prices.view(), window).unwrap();
        assert_eq!(result, array![2.0, 3.0, 4.0]);

        // Test case with not enough data
        let prices = array![1.0, 2.0];
        let window = 3;
        let result = calculate_ema(prices.view(), window);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::NotEnoughData(_)) => (),
            _ => {
                panic!("Expected `IndicatorError::NotEnoughData`, found different `IndicatorError`")
            }
        }
    }

    #[test]
    fn test_calculate_macd() {
        // Test case with enough data
        let prices = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let short_window = 2;
        let long_window = 4;
        let signal_window = 2;
        let result =
            calculate_macd(prices.view(), short_window, long_window, signal_window).unwrap();
        assert_eq!(result, (array![1.0], array![1.0], array![0.0]));

        // Test case with not enough data
        let prices = array![1.0, 2.0];
        let short_window = 2;
        let long_window = 4;
        let signal_window = 2;
        let result = calculate_macd(prices.view(), short_window, long_window, signal_window);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::NotEnoughData(_)) => (),
            _ => {
                panic!("Expected `IndicatorError::NotEnoughData`, found different `IndicatorError`")
            }
        }
    }
}
