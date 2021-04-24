import pyfirmata as fm
from loguru import logger
class ArduinoAdapter:

    def __init__(self, port: str = "COM5") -> None:
        self.board = self.__setup_board(port)
        
    def __setup_board(self, port: str) -> fm.Arduino:
        board = None
        try:
            teensy36 = {
                'digital' : tuple(x for x in range(40)),
                'analog' : tuple(x for x in range(23)),
                'pwm' : (2,3,4,5,6,7,8,9,10,14,16,17,20,21,22,23,29,30,35,36,37,38),
                'use_ports' : True,
                'disabled' : (0, 1) # Rx, Tx, Crystal 
            }
            board = fm.Arduino(port)
            board.digital[5].mode = fm.OUTPUT
            board.setup_layout(teensy36)
        except Exception:
            logger.warning(f"Arduino Board not found.")
        finally:
            return board

    def set_led(self, mask: bool) -> None:
        if mask is True:
            self.board.digital[5].write(1)
        else:
            self.board.digital[5].write(0)