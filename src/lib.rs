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
/// * `prices` - A slice of price data.
/// * `window` - The size of the window for calculating RSI.
///
/// # Returns
///
/// A Result containing a vector of RSI values or an `IndicatorError` if there is not enough data.
///
/// # Errors
///
/// Returns an `IndicatorError::NotEnoughData` if the length of `prices` is less than or equal to
/// `window`.
pub fn calculate_rsi(prices: &[f64], window: usize) -> Result<Vec<f64>, IndicatorError> {
    // Check if prices array has enough elements
    if prices.len() <= window {
        return Err(IndicatorError::NotEnoughData(
            "Not enough data points to calculate RSI".to_string(),
        ));
    }

    // Calculate price changes
    let price_changes = prices[1..].iter().zip(prices.iter()).map(|(x, y)| x - y);

    // Separate gains and losses
    let gains: Vec<f64> = price_changes
        .clone()
        .map(|x| if x > 0.0 { x } else { 0.0 })
        .collect();
    let losses: Vec<f64> = price_changes
        .map(|x| if x < 0.0 { -x } else { 0.0 })
        .collect();

    // Calculate average gains and losses over the window
    let mut avg_gain = gains.iter().take(window).sum::<f64>() / window as f64;
    let mut avg_loss = losses.iter().take(window).sum::<f64>() / window as f64;

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

    Ok(rsi_values)
}

/// Calculates the Exponential Moving Average (EMA) for a given price array and window size.
///
/// # Arguments
///
/// * `prices` - A slice of price data.
/// * `window` - The size of the window for calculating EMA.
///
/// # Returns
///
/// A Result containing a vector of EMA values or an `IndicatorError` if there is not enough data.
///
/// # Errors
///
/// Returns an `IndicatorError::NotEnoughData` if the length of `prices` is less than `window`.
pub fn calculate_ema(prices: &[f64], window: usize) -> Result<Vec<f64>, IndicatorError> {
    if prices.len() < window {
        return Err(IndicatorError::NotEnoughData(
            "`prices` must have at least `window` items".to_string(),
        ));
    }

    let smoothing = 2.0 / (window as f64 + 1.0);

    let sma = prices.iter().take(window).sum::<f64>() / window as f64;
    let mut ema_values = Vec::with_capacity(prices.len() - window);
    ema_values.push(sma);

    for i in window..prices.len() {
        let current_price = prices[i];
        let prev_ema = ema_values[i - window];

        let ema = (current_price - prev_ema) * smoothing + prev_ema;
        ema_values.push(ema);
    }

    Ok(ema_values)
}

/// Calculates the Moving Average Convergence Divergence (MACD) for a given price array and parameters.
///
/// # Arguments
///
/// * `prices` - A slice of price data.
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
    prices: &[f64],
    short_window: usize,
    long_window: usize,
    signal_window: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), IndicatorError> {
    let mut ema_short = calculate_ema(prices, short_window)?;
    let ema_long = calculate_ema(prices, long_window)?;
    ema_short = ema_short[long_window - short_window..].to_owned();

    let mut macd_line = ema_short
        .iter()
        .zip(&ema_long)
        .map(|(a, b)| a - b)
        .collect::<Vec<f64>>();
    let signal_line = calculate_ema(&macd_line, signal_window)?;
    macd_line = macd_line[macd_line.len() - signal_line.len()..].to_owned();

    let histogram = macd_line
        .clone()
        .iter()
        .zip(&signal_line)
        .map(|(a, b)| a - b)
        .collect::<Vec<f64>>();
    Ok((macd_line, signal_line, histogram))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_rsi() {
        // Test case with enough data
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let window = 3;
        let result = calculate_rsi(prices.as_slice(), window).unwrap();
        assert_eq!(result, vec![100.0, 100.0]);

        // Test case with not enough data
        let prices = vec![1.0, 2.0];
        let window = 3;
        let result = calculate_rsi(prices.as_slice(), window);
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
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let window = 3;
        let result = calculate_ema(prices.as_slice(), window).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0]);

        // Test case with not enough data
        let prices = vec![1.0, 2.0];
        let window = 3;
        let result = calculate_ema(prices.as_slice(), window);
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
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let short_window = 2;
        let long_window = 4;
        let signal_window = 2;
        let result =
            calculate_macd(prices.as_slice(), short_window, long_window, signal_window).unwrap();
        assert_eq!(result, (vec![1.0], vec![1.0], vec![0.0]));

        // Test case with not enough data
        let prices = vec![1.0, 2.0];
        let short_window = 2;
        let long_window = 4;
        let signal_window = 2;
        let result = calculate_macd(prices.as_slice(), short_window, long_window, signal_window);
        assert!(result.is_err());
        match result {
            Err(IndicatorError::NotEnoughData(_)) => (),
            _ => {
                panic!("Expected `IndicatorError::NotEnoughData`, found different `IndicatorError`")
            }
        }
    }
}
