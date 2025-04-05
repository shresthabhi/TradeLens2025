import pandas as pd

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
    

    from pandas import date_range

def get_required_period_keys(start_date, end_date, data_type, frequency, rules):
    """
    Returns a list of chunk keys (e.g., ['2023', '2024'] or ['2023-01', '2023-02'])
    that might contain data between start_date and end_date.
    """
    fmt = rules[data_type][frequency]["file_format"]

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    if fmt == "%Y":
        periods = pd.date_range(start, end, freq="Y")
    elif fmt == "%Y-%m":
        periods = pd.date_range(start, end, freq="MS")
    elif fmt == "%Y-Q%q":
        # Special case — quarter calculation
        dates = pd.date_range(start, end, freq="Q")
        periods = [f"{d.year}-Q{((d.month - 1)//3 + 1)}" for d in dates]
        return sorted(set(periods))
    else:
        return ["static"]

    return sorted({date.strftime(fmt) for date in periods})
