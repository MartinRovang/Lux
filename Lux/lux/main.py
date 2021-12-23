from lux import PriceFetcher

fetcher = PriceFetcher()
print(fetcher.fetch_info('AFK.OL', (2021,1,1), (2021,5,1)))
