import os

# Path to the app root directory
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# Final data folder path
# DATA_ROOT = os.path.join(APP_ROOT, "data")
DATA_ROOT = os.path.join("/tmp/", "data")

if not os.path.exists(DATA_ROOT):
    os.makedirs(DATA_ROOT)

# Frequency config for file handling
STORAGE_RULES = {
    "price_data": {
        "daily": {
            "split_by": "year",
            "file_format": "%Y"
        },
        "weekly": {
            "split_by": "year",
            "file_format": "%Y"
        },
        "intraday_1min": {
            "split_by": "month",
            "file_format": "%Y-%m"
        }
    },
    "financials": {
        "quarterly": {
            "split_by": "quarter",
            "file_format": "%Y-Q%q"  # %q is not built-in; we’ll define it
        },
        "yearly": {
            "split_by": "year",
            "file_format": "%Y"
        }
    },
    "analyst_ratings": {
        "monthly": {
            "split_by": "month",
            "file_format": "%Y-%m"
        }
    },
    "company_info": {
        "static": {
            "split_by": None,
            "file_format": None
        }
    }
}
