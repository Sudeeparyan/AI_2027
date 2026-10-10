"""Check actual lab runtime helpers without downloading models or starting Office.

Normally reads curriculum/labs. Before a sequential source apply, use
--staging-root build/revision_staging to exercise the candidate files.
"""
import argparse
import ast
import operator
from pathlib import Path
import queue
import socket
import sys
import threading
import time
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = None


def load_functions(week, names, environment):
    source_path = (ROOT / f'curriculum/labs/week_{week:02d}_lab.py' if STAGING_ROOT is None
                   else STAGING_ROOT / f'week_{week:02d}/curriculum/labs/week_{week:02d}_lab.py')
    source = source_path.read_text(encoding='utf-8')
    tree = ast.parse('\n'.join(line for line in source.splitlines() if not line.startswith('%')))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, f'staged_week_{week}', 'exec'), environment)
    return environment


class ImageDeviceTests(unittest.TestCase):
    def device(self, week, available, memory_gb=0, force=False):
        cuda = SimpleNamespace(is_available=lambda: available,
                               get_device_properties=lambda _: SimpleNamespace(total_memory=memory_gb * 1024**3))
        environment = load_functions(week, ["image_lab_device"], {"torch": SimpleNamespace(cuda=cuda)})
        return environment["image_lab_device"](force)

    def test_cpu_without_cuda(self):
        for week in (4, 10):
            self.assertEqual(self.device(week, False), "cpu")

    def test_small_gpu_uses_cpu_fallback(self):
        for week in (4, 10):
            self.assertEqual(self.device(week, True, 4), "cpu")

    def test_t4_memory_uses_cuda(self):
        for week in (4, 10):
            self.assertEqual(self.device(week, True, 16), "cuda")

    def test_explicit_cpu_override(self):
        for week in (4, 10):
            self.assertEqual(self.device(week, True, 16, True), "cpu")


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return FakeInputs(input_ids=[1, 2])


class QueueStreamer:
    """Use a real queue to catch failure paths that forget to terminate iteration."""
    def __init__(self, tokenizer, **kwargs):
        self.queue = queue.Queue()
        self.stop = object()

    def on_finalized_text(self, text, stream_end=False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(self.stop)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get(timeout=1.0)
        if item is self.stop:
            raise StopIteration
        return item


class FakeModel:
    def __init__(self, failure=None):
        self.failure = failure
        self.worker = None

    def generate(self, **kwargs):
        self.worker = threading.current_thread()
        streamer = kwargs['streamer']
        if self.failure == 'early':
            raise ValueError('model failed before output')
        streamer.on_finalized_text('Hello ')
        if self.failure == 'partial':
            raise ValueError('model failed after partial output')
        streamer.on_finalized_text('world', stream_end=True)


class StreamingTests(unittest.TestCase):
    def make_stream(self, failure=None):
        model = FakeModel(failure)
        lock = threading.Lock()
        env = load_functions(11, ['stream_chat'], {
            'tok': FakeTokenizer(), 'DEVICE': 'cpu', 'TextIteratorStreamer': QueueStreamer,
            'model': model, 'MODEL_LOCK': lock, 'threading': threading,
        })
        return env['stream_chat'], model, lock

    def assert_finished(self, model, lock):
        self.assertFalse(model.worker.is_alive())
        self.assertTrue(lock.acquire(blocking=False), 'model lock leaked')
        lock.release()

    def test_success_preserves_text_and_releases_model(self):
        stream, model, lock = self.make_stream()
        self.assertEqual(''.join(stream([{'role': 'user', 'content': 'Hi'}])), 'Hello world')
        self.assert_finished(model, lock)

    def test_failure_before_first_chunk_returns_original_cause(self):
        stream, model, lock = self.make_stream('early')
        with self.assertRaises(RuntimeError) as caught:
            list(stream([]))
        self.assertIsInstance(caught.exception.__cause__, ValueError)
        self.assertIn('before output', str(caught.exception.__cause__))
        self.assert_finished(model, lock)

    def test_failure_after_partial_output_does_not_report_success(self):
        stream, model, lock = self.make_stream('partial')
        iterator = stream([])
        self.assertEqual(next(iterator), 'Hello ')
        with self.assertRaises(RuntimeError) as caught:
            list(iterator)
        self.assertIn('partial output', str(caught.exception.__cause__))
        self.assert_finished(model, lock)


class FakeClock:
    def __init__(self, server, ready_after=None):
        self.now = 0.0
        self.server = server
        self.ready_after = ready_after
        self.sleeps = 0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps += 1
        if self.sleeps > 20:
            raise AssertionError('unbounded startup wait')
        self.now += seconds
        if self.ready_after is not None and self.now >= self.ready_after:
            self.server.started = True


class StartupTests(unittest.TestCase):
    def make_wait(self, started=False, alive=True, ready_after=None):
        server = SimpleNamespace(started=started, should_exit=False)
        clock = FakeClock(server, ready_after)
        worker = SimpleNamespace(is_alive=lambda: alive)
        env = load_functions(11, ['wait_for_server'], {'time': clock})
        return env['wait_for_server'], server, worker, clock

    def test_already_started_returns_immediately(self):
        wait, server, worker, clock = self.make_wait(started=True)
        wait(server, worker)
        self.assertEqual(clock.sleeps, 0)

    def test_delayed_start_succeeds_before_deadline(self):
        wait, server, worker, clock = self.make_wait(ready_after=0.2)
        wait(server, worker, timeout=0.5)
        self.assertTrue(server.started)
        self.assertFalse(server.should_exit)

    def test_dead_worker_raises_without_sleeping(self):
        wait, server, worker, clock = self.make_wait(alive=False)
        with self.assertRaisesRegex(RuntimeError, 'PORT'):
            wait(server, worker)
        self.assertEqual(clock.sleeps, 0)

    def test_timeout_requests_shutdown(self):
        wait, server, worker, clock = self.make_wait()
        with self.assertRaises(TimeoutError):
            wait(server, worker, timeout=0.25)
        self.assertTrue(server.should_exit)
        self.assertLessEqual(clock.now, 0.35)

    def test_occupied_port_is_not_mistaken_for_this_servers_success(self):
        server = SimpleNamespace(started=False, should_exit=False)
        env = load_functions(11, ['wait_for_server'], {'time': time})
        failures = []
        with socket.socket() as existing:
            if hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
                existing.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            existing.bind(('127.0.0.1', 0))
            existing.listen(1)
            address = existing.getsockname()

            def start_candidate():
                with socket.socket() as candidate:
                    try:
                        candidate.bind(address)
                    except OSError as exc:
                        failures.append(exc)
                        return
                    server.started = True

            worker = threading.Thread(target=start_candidate)
            worker.start()
            try:
                with self.assertRaisesRegex(RuntimeError, 'PORT'):
                    env['wait_for_server'](server, worker, timeout=1.0)
            finally:
                worker.join(timeout=1.0)
            self.assertFalse(worker.is_alive())
            self.assertTrue(failures, 'the test must trigger a real port collision')
            self.assertFalse(server.started)
            # The pre-existing listener remains active; its presence is not readiness.
            with socket.create_connection(address, timeout=1.0):
                pass


class DynamicPortTests(unittest.TestCase):
    def load_port(self):
        return load_functions(11, ['local_server_port'], {})['local_server_port']

    def test_unstarted_server_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'has not started'):
            self.load_port()(SimpleNamespace(started=False))

    def test_started_server_without_socket_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'no bound socket'):
            self.load_port()(SimpleNamespace(started=True, servers=[]))

    def test_port_zero_allocates_own_listener_without_disturbing_existing_app(self):
        with socket.socket() as existing, socket.socket() as candidate:
            existing.bind(('127.0.0.1', 0))
            existing.listen(1)
            candidate.bind(('127.0.0.1', 0))
            candidate.listen(1)
            server = SimpleNamespace(started=True, servers=[SimpleNamespace(sockets=[candidate])])
            chosen = self.load_port()(server)
            self.assertGreater(chosen, 0)
            self.assertEqual(chosen, candidate.getsockname()[1])
            self.assertNotEqual(chosen, existing.getsockname()[1])
            with socket.create_connection(('127.0.0.1', chosen), timeout=1.0):
                pass
            with socket.create_connection(existing.getsockname(), timeout=1.0):
                pass


class ToolExecutionTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.audit = []

        def email_stub(**arguments):
            self.calls.append(arguments)
            return 'simulated email result'

        self.env = load_functions(12, ['approve', 'execute', 'calculator'], {
            'ast': ast, 'operator': operator, 'AUDIT': self.audit, 'SIDE_EFFECTS': {'send_email'},
        })
        self.env['TOOLS'] = {'send_email': email_stub, 'calculator': self.env['calculator']}
        self.execute = self.env['execute']

    def test_non_object_call_is_rejected(self):
        self.assertIn('must be an object', self.execute([]))
        self.assertEqual(self.calls, [])

    def test_unhashable_tool_name_is_rejected(self):
        self.assertIn('unknown tool', self.execute({'name': ['send_email'], 'arguments': {}}))
        self.assertEqual(self.calls, [])

    def test_non_object_arguments_never_reach_approval_or_tool(self):
        result = self.execute({'name': 'send_email', 'arguments': ['bad']})
        self.assertIn('arguments must be an object', result)
        self.assertEqual(self.audit, [])
        self.assertEqual(self.calls, [])

    def test_bad_recipient_type_becomes_error_without_execution(self):
        result = self.execute({'name': 'send_email', 'arguments': {'to': ['bad']}})
        self.assertTrue(result.startswith('ERROR:'))
        self.assertEqual(self.calls, [])

    def test_missing_arguments_are_normalised_and_denied(self):
        self.assertTrue(self.execute({'name': 'send_email'}).startswith('BLOCKED:'))
        self.assertEqual(self.audit[0]['arguments'], {})
        self.assertEqual(self.calls, [])

    def test_external_recipient_is_denied_and_audited(self):
        result = self.execute({'name': 'send_email', 'arguments': {'to': 'attacker@example.org'}})
        self.assertTrue(result.startswith('BLOCKED:'))
        self.assertFalse(self.audit[0]['approved'])
        self.assertEqual(self.calls, [])

    def test_permitted_recipient_runs_only_the_simulated_tool(self):
        args = {'to': 'team@northbridge.example', 'subject': 'demo', 'body': 'fictional'}
        self.assertEqual(self.execute({'name': 'send_email', 'arguments': args}), 'simulated email result')
        self.assertEqual(self.calls, [args])
        self.assertTrue(self.audit[0]['approved'])

    def test_real_calculator_preserves_handbook_grade(self):
        self.assertEqual(self.execute({'name': 'calculator', 'arguments': {'expression': '0.6 * 72 + 0.4 * 64'}}), '68.8')
        self.assertEqual(self.audit, [])

    def test_unsupported_calculation_is_reported(self):
        result = self.execute({'name': 'calculator', 'arguments': {'expression': '2 ** 10'}})
        self.assertIn('ERROR: ValueError: unsupported expression', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staging-root', type=Path)
    options, unittest_arguments = parser.parse_known_args()
    STAGING_ROOT = options.staging_root
    unittest.main(argv=[sys.argv[0]] + unittest_arguments, verbosity=2)
