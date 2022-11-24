import logging
import pytest
import re

def tpu_perf_run(nntc_env, cmd):
    if not nntc_env['case_list']:
        logging.info(f'Skip {cmd}')
        return

    container = nntc_env['nntc_container']
    cmd = f'{cmd} {nntc_env["case_list"]}'
    ret, output = container.exec_run(
        f'bash -c "{cmd}"',
        tty=True)
    output = output.decode()
    logging.info(f'{cmd}\n{output}')
    m = re.search('(?<=please check ).+\\.log', output)
    if m:
        log_fn = m.group(0).replace('/workspace', '.')
        with open(log_fn) as f:
            logging.info(f'cat {log_fn}\n{f.read()}')

    assert ret == 0

@pytest.fixture(scope='module')
def test_efficiency(nntc_env):
    tpu_perf_run(nntc_env, 'python3 -m tpu_perf.build --time')

@pytest.mark.usefixtures('test_efficiency')
def test_accuracy(nntc_env):
    tpu_perf_run(nntc_env, 'python3 -m tpu_perf.build')
