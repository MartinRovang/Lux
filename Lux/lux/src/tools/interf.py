import rich
from rich.console import Console
from rich.table import Table

class Interface:
    def __init__(self):
        self.console = Console()
    
    def make_table_moments(self):
        self.table = Table(title="Portofolio")
        self.table.add_column("Ticker", style="cyan", no_wrap=True)
        self.table.add_column("μθ30wg10", style="cyan", no_wrap=True, justify="center")
        # self.table.add_column("μb30wg10", style="cyan", no_wrap=True, justify="center")
        self.table.add_column("μwg10 now", style="cyan", no_wrap=True, justify="center")
        self.table.add_column("zσψ30wg10", style="cyan", no_wrap=True, justify="center")
        # self.table.add_column("zσb30wg10", style="cyan", no_wrap=True, justify="center")
        self.table.add_column("zσwg10 now", style="cyan", no_wrap=True, justify="center")
        self.table.add_column("zσwg10 min/max|baseline.", style="cyan", no_wrap=True, justify="center")
    
    def add_row_moments(self, moments, ticker):
        for moment in moments:
            if moments[moment]['color'] == 'green/red':
                if moments[moment]['value'] > 0:
                    color_output = 'green'
                else:
                    color_output = 'red'
            else:
                color_output = moments[moment]['color']
            moments[moment]['output_text'] = f"[bold {color_output}] {moments[moment]['value']:.5f} [/bold {color_output}] "
        # self.table.add_row(ticker, moments['mean_coeff']['output_text'] + "R^2"+moments['scoremean']['output_text'], moments['mean_intercept']['output_text'], moments['mean_now']['output_text'], moments['var_coeff']['output_text'] + "R^2"+moments['scorevar']['output_text'], moments['var_intercept']['output_text'], moments['var_now']['output_text'], f"[{moments['minimum_historic_std']['output_text']}/{moments['maximum_historic_std']['output_text']}|{moments['baseline_historic_std']['output_text']}]")
        self.table.add_row(ticker, moments['mean_coeff']['output_text'] + "R^2"+moments['scoremean']['output_text'], moments['mean_now']['output_text'], moments['var_coeff']['output_text'] + "R^2"+moments['scorevar']['output_text'], moments['var_now']['output_text'], f"[{moments['minimum_historic_std']['output_text']}/{moments['maximum_historic_std']['output_text']}|{moments['baseline_historic_std']['output_text']}]")


    def console_print(self):
        self.console.print(self.table)