# for setting our open and close times
from datetime import time
# for setting our start and end sessions
# import pandas as pd
# for setting which days of the week we trade on
from pandas.tseries.offsets import CustomBusinessDay
# for setting our timezone
from pytz import timezone

# for creating and registering our calendar
# from trading_calendars import register_calendar, TradingCalendar
from zipline.utils.memoize import lazyval
from zipline.utils.calendars import TradingCalendar


class TFSExchangeCalendar(TradingCalendar):
    """
    An exchange calendar for trading assets 24/7.

    Open Time: 12AM, UTC
    Close Time: 11:59PM, UTC
    """

    @property
    def name(self):
        """
        The name of the exchange, which Zipline will look for
        when we run our algorithm and pass TFS to
        the --trading-calendar CLI flag.
        """
        return "TFS"

    @property
    def tz(self):
        """
        The timezone in which we'll be running our algorithm.
        """
        return timezone("UTC")

    @property
    def open_time(self):
        """
        The time in which our exchange will open each day.
        """
        return time(0, 0)

    @property
    def close_time(self):
        """
        The time in which our exchange will close each day.
        """
        return time(23, 59)

    @lazyval
    def day(self):
        """
        The days on which our exchange will be open.
        """
        weekmask = "Mon Tue Wed Thu Fri Sat Sun"
        return CustomBusinessDay(
            weekmask=weekmask
        )
