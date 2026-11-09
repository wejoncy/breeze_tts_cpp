"""Native HTTP integration checks for Engine's bidirectional request queues."""
import argparse
import base64
import concurrent.futures
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import wave


def check(args):
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    build = str(Path(args.build).resolve())
    command = [str(Path(build) / 'tts-server'), '--model', args.model,
               '--host', '127.0.0.1', '--port', str(port), '--max-batch', '4',
               '--context', '2048', '--quantization', 'int8', '--runtime-memory-mb', '2048',
               '--memory-limit-mb', '12288', '--queue-limit', '32', '--http-threads', '16',
               '--log-level', 'info']
    report = Path(args.output)
    report.parent.mkdir(parents=True, exist_ok=True)
    options = dict(input='你好，今天我们沿着河边散步，看看小镇的风景。',
                   response_format='pcm', max_frames=8, cfg_scale=1, temperature=0,
                   depth_temperature=0, do_sample=False, depth_do_sample=False, seed=42)

    def post(payload, path='/v1/audio/speech'):
        conn = http.client.HTTPConnection('127.0.0.1', port, timeout=60)
        try:
            conn.request('POST', path, json.dumps(payload), {'Content-Type': 'application/json'})
            response = conn.getresponse()
            body = response.read()
            if response.status != 200:
                raise RuntimeError((response.status, body.decode()))
            return body, dict(response.getheaders())
        finally:
            conn.close()

    log_path = report.with_suffix('.server.log')
    with log_path.open('w') as log:
        process = subprocess.Popen(command, env=dict(os.environ, LD_LIBRARY_PATH=build),
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 240
            while True:
                if process.poll() is not None:
                    raise RuntimeError('server exited during startup')
                conn = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                try:
                    conn.request('GET', '/health')
                    response = conn.getresponse()
                    if response.status == 200:
                        response.read()
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('server startup timeout')
                    time.sleep(.1)
                finally:
                    conn.close()

            pcm, headers = post(dict(options, stream=True))
            assert pcm and headers['Content-Type'] == 'audio/pcm'
            offline, _ = post(dict(options, stream=False))
            assert pcm == offline, 'stream/nonstream output queue mismatch'
            sse, _ = post(dict(options, stream_format='sse'))
            events = [json.loads(line[6:]) for line in sse.decode().splitlines() if line.startswith('data: ')]
            assert events[-1]['type'] == 'speech.audio.done', 'missing terminal queue message'
            deltas = [base64.b64decode(event['audio']) for event in events if event['type'] == 'speech.audio.delta']
            assert b''.join(deltas) == pcm, 'SSE output queue mismatch'
            wav, _ = post(dict(options, response_format='wav'))
            with wave.open(io.BytesIO(wav)) as audio:
                assert audio.readframes(audio.getnframes()) == pcm, 'WAV output queue mismatch'

            # A disconnected long request must cancel without stranding other clients.
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=60)
            conn.request('POST', '/v1/audio/speech', json.dumps(dict(options, stream=True, max_frames=300)),
                         {'Content-Type': 'application/json'})
            response = conn.getresponse()
            assert response.status == 200 and response.read(2)
            response.close()
            conn.close()
            with concurrent.futures.ThreadPoolExecutor(max_workers=9) as pool:
                futures = [pool.submit(post, dict(options, stream=True, max_frames=5+i,
                    input=f'第{i+1}篇记录。'+options['input'])) for i in range(9)]
                concurrent_outputs = [future.result() for future in futures]
            ids = [item[1]['X-Request-Id'] for item in concurrent_outputs]
            assert len(set(ids)) == 9, 'HTTP identities collided'
            samples_per_frame = int(headers['X-Samples-Per-Frame'])
            for i, (audio, _) in enumerate(concurrent_outputs):
                assert len(audio) == (5+i) * samples_per_frame * 2, 'HTTP worker received another request output'

            batch, _ = post(dict(options, items=[dict(input='第一项，今天阳光明媚。', max_frames=5),
                dict(input='第二项，我们准备出发。', max_frames=7), dict(input='')]), '/v1/audio/speech/batch')
            batch = json.loads(batch)
            assert batch['succeeded'] == 2 and batch['failed'] == 1, 'batch error isolation failed'
            for index, frames in enumerate((5,7)):
                result = batch['results'][index]
                assert result['status'] == 'success'
                assert len(base64.b64decode(result['audio_data'])) == frames * samples_per_frame * 2
            assert post(dict(options))[0] == pcm, 'disconnect or invalid batch poisoned Engine'
            result = dict(command=command, checks=['streaming PCM', 'nonstream PCM', 'SSE deltas and terminal',
                'WAV framing', 'disconnect cancellation', 'nine concurrent output queues', 'batch error isolation'],
                greedy_pcm_sha256=hashlib.sha256(pcm).hexdigest(),
                concurrent_request_ids=ids, concurrent_audio_bytes=[len(item[0]) for item in concurrent_outputs])
        finally:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    assert 'cancelled=true' in log_path.read_text(), 'disconnected request was not cancelled'
    report.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', required=True)
    parser.add_argument('--model', default='/workspace/Breeze-TTS-2')
    parser.add_argument('--output', required=True)
    print(json.dumps(check(parser.parse_args()), indent=2))
