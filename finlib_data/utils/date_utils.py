import pandas as pd
from calendar import monthrange

def get_time_key_for_frequency(date, data_type, frequency, rules):
    """
    Given a date, return the storage chunk key for a given data type + frequency.
    Example: '2023', '2023-01', '2022-Q3'
    """
    fmt = rules[data_type][frequency]["file_format"]
    if fmt == "%Y-Q%q":
        year = date.year
        quarter = (date.month - 1) // 3 + 1
        return f"{year}-Q{quarter}"
    elif fmt:
        return date.strftime(fmt)
    else:
        return "static"  # for company info etc.
    

def get_date_range_for_key(key, data_type, frequency, rules):
    fmt = rules[data_type][frequency]["file_format"]

    if fmt == "%Y":
        year = int(key)
        min_date = pd.Timestamp(f"{year}-01-01")
        max_date = pd.Timestamp(f"{year}-12-31")

    elif fmt == "%Y-%m":
        year, month = map(int, key.split("-"))
        min_date = pd.Timestamp(f"{year}-{month:02d}-01")
        last_day = monthrange(year, month)[1]
        max_date = pd.Timestamp(f"{year}-{month:02d}-{last_day}")

    elif fmt == "%Y-Q%q":
        year_str, qtr_str = key.split("-Q")
        year = int(year_str)
        quarter = int(qtr_str)
        start_month = 3 * (quarter - 1) + 1
        end_month = start_month + 2
        min_date = pd.Timestamp(f"{year}-{start_month:02d}-01")
        last_day = monthrange(year, end_month)[1]
        max_date = pd.Timestamp(f"{year}-{end_month:02d}-{last_day}")

    elif fmt is None:
        min_date = max_date = None  # no date range for static chunks

    else:
        raise NotImplementedError(f"Unsupported format: {fmt}")

    return min_date, max_date


def get_required_period_keys(start_date, end_date, data_type, frequency, rules):
    """
    Returns a list of chunk keys (e.g., ['2023', '2024'] or ['2023-01', '2023-02'])
    that might contain data between start_date and end_date.
    """
    fmt = rules[data_type][frequency]["file_format"]

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    if fmt == "%Y":
        years = list(range(start.year, end.year + 1))
        return [str(year) for year in years]

    elif fmt == "%Y-%m":
        periods = pd.date_range(start, end, freq="MS")
        if len(periods) == 0:  # ensure at least the start month
            periods = [start.replace(day=1)]
        return sorted({date.strftime(fmt) for date in periods})

    elif fmt == "%Y-Q%q":
        # Pad start and end to quarter boundaries
        start_q = pd.Period(start, freq='Q')
        end_q = pd.Period(end, freq='Q')
        quarters = pd.period_range(start_q, end_q, freq='Q')
        return [f"{q.year}-Q{q.quarter}" for q in quarters]

    else:
        return ["static"]

# def get_required_period_keys(start_date, end_date, data_type, frequency, rules):
#     """
#     Returns a list of chunk keys (e.g., ['2023', '2024'] or ['2023-01', '2023-02'])
#     that might contain data between start_date and end_date.
#     """
#     fmt = rules[data_type][frequency]["file_format"]

#     start = pd.to_datetime(start_date)
#     end = pd.to_datetime(end_date)

#     if fmt == "%Y":
#         periods = pd.date_range(start, end, freq="Y")
#     elif fmt == "%Y-%m":
#         periods = pd.date_range(start, end, freq="MS")
#     elif fmt == "%Y-Q%q":
#         # Special case — quarter calculation
#         dates = pd.date_range(start, end, freq="QE")
#         periods = [f"{d.year}-Q{((d.month - 1)//3 + 1)}" for d in dates]
#         return sorted(set(periods))
#     else:
#         return ["static"]

#     return sorted({date.strftime(fmt) for date in periods})
