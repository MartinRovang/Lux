import rich
from rich.console import Console
from rich.table import Table
class Interface:
    def __init__(self):
        self.console = Console()

    
    def colorify(self, text, color, type = 'float'):
      if type == 'float':
        output = f"[bold {color}] {text:.3f} [/bold {color}]"
      else:
        output = f"[bold {color}] {text} [/bold {color}]"
      return output
    
    def redgreencolor(self, value):
        if value > 0:
            return f"[bold green] {value:.5f} [/bold green] "
        else:
            return f"[bold red] {value:.5f} [/bold red] "

    
    def make_table(self, col, tl = 'Portofolio'):
      self.table = Table(title = tl)
      for key in col:
        self.table.add_column(key, style="cyan", no_wrap=True, justify="center")
    
    def add_row(self, rows):
      self.table.add_row(*rows)


    def console_print(self):
        self.console.print(self.table)
    
    def print_regular(self, text, color):
        self.console.print(f"[bold {color}] {text} [/bold {color}]")


    def show_logo(self):
        logo = """
        [bold cyan]                                                  
                      *@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                  
                     @@@@@&&&&&&&&@@@@@@&&&&&&&&&&&&&&@@@@@&&&&&&&&&@@@@&               
                  &@@@@@        @@@@@(       &@@@@%     %@@@@@        @@@@@#            
               (@@@@@*       &@@@@&            *@@@@@/     &@@@@&       /@@@@@*         
              @@@@@        (@@@@%                *@@@@@      &@@@@*       *@@@@@        
                &@@@@/        @@@@@            (@@@@%      @@@@@        #@@@@#          
                  (@@@@%        &@@@@*       &@@@@/     (@@@@&        @@@@@*            
                     @@@@@        #@@@@#   @@@@@      &@@@@/        @@@@@               
                       &@@@@*       *@@@@&  &&      @@@@@        (@@@@%                 
                         #@@@@%        @@@@@     /@@@@&        &@@@@/                   
                           *@@@@@        %@@@@(%@@@@#        @@@@@                      
                              @@@@@        /@@@@@@*       /@@@@&                        
                                %@@@@(    *@@@@@@@@     %@@@@#                          
                                  /@@@@&(@@@@%  &@@@@/@@@@@                             
                                     @@@@@@/      (@@@@@&                               
                                       &@           *@#                                 
                                                                                        
                                                                                        
                              /////                                                     
                                #                                                       
                                #           @    %*       &  @                          
                                #     @     @    %         @@                           
                              &&@&&&&&&      @%&%(&&    &@&  &@&

                                  Made by Martin Soria Røvang
                                 Mail: Gimpedillusion@gmail.com
                                 Github: github.com/Martinrovang
                                    Instagram: @martin.rovang
        [/bold cyan]"""
        self.console.print(logo)