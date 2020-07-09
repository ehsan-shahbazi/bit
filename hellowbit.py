from binance.client import Client
import time
a = 1593688613


def wait_until(time_stamp, secs=10):
    time_stamp = int(time_stamp)
    sleep = int(60 - ((time_stamp/60) - int(time_stamp/60)) * 60) - secs
    if sleep > 0:
        time.sleep(sleep)
    else:
        time.sleep(sleep + 60)
    return True
