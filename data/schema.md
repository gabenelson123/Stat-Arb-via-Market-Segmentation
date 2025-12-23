# Dataset Schema

The project uses a wide-format daily price panel derived from Bloomberg-style
equity identifiers.

## Structure

- Rows: Trading dates
- Columns: Individual assets
- Values: Closing Price

## Format

| Column | Description |
|------|------------|
| Dates | Trading date (YYYY-MM-DD) |
| TICKER | Daily last price for asset |

Example tickers:
- A UN Equity
- AAPL UW Equity
- AA UN Equity
- AAL UW Equity

## Date Range

- Start: 2010-01-04
- End: 2025-09-12

## Notes
- The word "Dates" is on A3 and dates begin A4
- Equities are listed across the first row.  No information on the second and third rows.
- Prices are assumed to be adjusted for corporate actions.
- Missing values may occur due to IPOs, delistings, or data gaps.
