from lux import portofolio_manager

p = portofolio_manager.Portofolio()
# p.add_ticker('AFK.OL')
p.add_ticker('SALM.OL')
# p.add_ticker('YAR.OL')
p.add_ticker('EQNR.OL')
p.add_ticker('TEL.OL')
p.add_ticker('MPCC.OL')
# p.add_ticker('MOWI.OL')
# p.get_stock_stat()
p.get_portofolio_stats()