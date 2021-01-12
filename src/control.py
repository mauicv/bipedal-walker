import serial


class Connection:
    def __init__(self):
        port = '/dev/ttyACM0'
        self.conn = serial.Serial(port, 9600, timeout=5)

    def send(self, data):
        data = str.encode(','.join([str(item) for item in data]))
        self.conn.write(data)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.conn.close()
