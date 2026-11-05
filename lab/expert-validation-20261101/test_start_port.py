"""Local ephemeral sockets only. No model/service starts or production port use."""
import errno
import importlib.util
from pathlib import Path
import socket
import unittest

source=Path(__file__).resolve().parent.parent/'deepseek41-bringup-20261101/start.py'
spec=importlib.util.spec_from_file_location('model_start',source)
start=importlib.util.module_from_spec(spec)
spec.loader.exec_module(start)


class PortTests(unittest.TestCase):
    def test_free_port_can_listen_and_is_released(self):
        with socket.socket() as port:
            port.bind(('127.0.0.1',0))
            number=port.getsockname()[1]
        start.check_api_bind('127.0.0.1',number)
        with socket.socket() as server:
            server.bind(('127.0.0.1',number))
            server.listen(1)

    def test_live_listener_still_rejected(self):
        for reuse_port in [False,True]:
            with self.subTest(reuse_port=reuse_port), socket.socket() as server:
                server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                if reuse_port:
                    server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEPORT,1)
                server.bind(('127.0.0.1',0));server.listen(1)
                with self.assertRaises(OSError) as error:
                    start.check_api_bind('127.0.0.1',server.getsockname()[1])
                self.assertEqual(error.exception.errno,errno.EADDRINUSE)

    def test_time_wait_reproduces_old_failure_but_allows_new_check(self):
        with socket.socket() as server, socket.socket() as client:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            server.bind(('127.0.0.1',0));server.listen(1)
            number=server.getsockname()[1]
            client.settimeout(2);client.connect(('127.0.0.1',number))
            connection,_=server.accept()
            with connection:
                connection.settimeout(2)
                connection.shutdown(socket.SHUT_WR)
                self.assertEqual(client.recv(1),b'')
                client.close()
                self.assertEqual(connection.recv(1),b'')
        with socket.socket() as legacy:
            with self.assertRaises(OSError) as error:
                legacy.bind(('127.0.0.1',number))
            self.assertEqual(error.exception.errno,errno.EADDRINUSE)
        start.check_api_bind('127.0.0.1',number)


if __name__=='__main__':unittest.main()
