"""Native HTTP streaming benchmark; retains per-request TTFT and frame times."""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import random
from pathlib import Path
import socket
import signal
import statistics
import subprocess
import threading
import time

def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)

PASSAGES = [
    '清晨的阳光穿过窗户，照在书桌旁。我们整理好行李，准备沿着河岸慢慢走到小镇，看看今天的集市。',
    '这次旅行没有安排紧张的日程。遇到喜欢的风景，就停下来拍照；发现安静的咖啡馆，就坐一会儿，听听当地人的故事。',
    '午后的图书馆很安静，窗外有树叶轻轻摇动。我找到一本讲述海洋的书，里面介绍了潮汐、珊瑚和深海生物的生活。',
    '厨房里传来饭菜的香味。大家一起洗菜、切菜，讨论晚餐要做什么。简单的一顿饭，因为有人分享，就变得格外温暖。',
    '学习一项新的技能，需要耐心，也需要反复练习。每天完成一个小目标，记录遇到的问题，几个月后就能看见自己的进步。',
    '城市正在修建新的公园，步道连接树林和草坪。孩子们可以在那里玩耍，老人可以散步，附近的居民也多了一个休息的地方。',
    '朋友带来一个有趣的问题：如果能回到过去，你会选择哪一天？有人想看历史上的大事，有人只想再和家人吃一顿普通的晚饭。',
    '傍晚，我们坐在山坡上等日落。天空从浅蓝变成金色，再慢慢染上粉红。远处的灯光亮起来，一天的旅程也接近了尾声。',
    '天气预报说明天可能下雨，所以我们提前检查了雨伞和外套。出门前留一点准备时间，往往能让一整天的安排更加从容。',
    '博物馆里展示着不同年代的生活用品。看着那些熟悉又陌生的物件，我们能够想象以前的人怎样工作、交流，又怎样度过节日。',
]

def workload(args, count):
    rng = random.Random(args.seed)
    jobs = []
    supplied = json.loads(Path(args.workload).read_text()) if args.workload else None
    for index in range(count):
        if supplied:
            job = dict(supplied[index % len(supplied)])
        elif args.varied:
            # Different content and length; max_frames is a budget, not a duration assertion.
            seconds = 10 + 90 * ((index * 7) % count) / max(1, count - 1)
            parts = rng.sample(PASSAGES, len(PASSAGES))
            target = int(seconds * 5)
            text = f'第{index + 1}篇生活记录。' + ''.join(parts)[:target] + '。'
            job = dict(text=text, max_frames=int(seconds / .08))
        else:
            job = dict(text=args.text, max_frames=args.frames)
        jobs.append(dict(index=index, seed=args.seed + index, **job))
    return jobs

def request(port, job, warmup=False, options=None, warmup_frames=24, run_start=None, arrival_at=None,
            launch_barrier=None):
    if arrival_at is not None:
        remaining = arrival_at - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=600)
    frames = min(job['max_frames'], warmup_frames) if warmup else job['max_frames']
    index, text = job['index'], job['text']
    payload = dict(input=text, response_format='pcm', stream=True,
                   max_frames=frames, cfg_scale=1, temperature=0,
                   depth_temperature=0, do_sample=False, depth_do_sample=False,
                   seed=job['seed'])
    payload.update(options or {})
    if launch_barrier is not None:
        launch_barrier.wait(timeout=30)
    start = time.perf_counter()
    conn.request('POST', '/v1/audio/speech', json.dumps(payload),
                 {'Content-Type': 'application/json'})
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError((response.status, response.read().decode()))
    if not response.chunked:
        raise RuntimeError('benchmark requires chunked audio')
    chunks = []
    pcm = bytearray()
    # Read exact HTTP chunks, preserving the native emission boundaries.
    while True:
        size = int(response.fp.readline().split(b';', 1)[0].strip(), 16)
        if not size:
            break
        chunk = response.fp.read(size)
        if len(chunk) != size or response.fp.read(2) != b'\r\n':
            raise RuntimeError('truncated HTTP chunk')
        pcm.extend(chunk)
        chunks.append(dict(at_ms=(time.perf_counter() - start) * 1000, bytes=size))
    total_ms = (time.perf_counter() - start) * 1000
    conn.close()
    if not chunks:
        raise RuntimeError('empty audio')
    sample_rate = int(response.getheader('X-Sample-Rate'))
    samples_per_frame = int(response.getheader('X-Samples-Per-Frame'))
    actual_frames = len(pcm) / (samples_per_frame * 2)
    if len(pcm) % 2:
        raise RuntimeError('incomplete PCM sample')
    first_frames = chunks[0]['bytes'] / (samples_per_frame * 2)
    return dict(index=index, started_at_ms=(start - run_start) * 1000 if run_start else None,
                ttft_ms=chunks[0]['at_ms'], total_ms=total_ms,
                tpot_ms=(chunks[-1]['at_ms'] - chunks[0]['at_ms']) / max(1, actual_frames - first_frames),
                frames=actual_frames, audio_seconds=len(pcm) / (2 * sample_rate),
                budget_frames=frames, sample_rate=sample_rate, samples_per_frame=samples_per_frame,
                sha256=hashlib.sha256(pcm).hexdigest(), chunks=chunks)

def run(args):
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    command = [str(Path(args.build) / 'tts-server'), '--model', args.model,
               '--host', '127.0.0.1', '--port', str(port), '--max-batch', str(args.batch),
               '--context', str(args.context), '--runtime-memory-mb', str(args.runtime_memory_mb), '--memory-limit-mb', '12288',
               '--queue-limit', '64', '--http-threads', '64', '--log-level', 'warn']
    if args.mode != 'bf16':
        command += ['--quantization', args.mode]
    jobs = workload(args, sum(args.concurrency) if args.single_wave else max(args.requests, max(args.concurrency) * 2))
    options = json.loads(Path(args.request_options).read_text()) if args.request_options else {}
    report = dict(build=args.build, mode=args.mode, command=command, frames=args.frames, workload=jobs,
                  request_options=options,
                  arrival_spacing_ms=args.arrival_spacing_ms,
                  simultaneous_start=args.simultaneous_start, single_wave=args.single_wave,
                  max_seconds=args.max_seconds,
                  batch_capacity=args.batch, text=args.text, rows=[])
    with out.with_suffix('.server.log').open('w') as log:
        process = subprocess.Popen(command, env=dict(os.environ, LD_LIBRARY_PATH=args.build),
                                   stdout=log, stderr=subprocess.STDOUT)
        def expired(signum, frame):
            process.kill()
            reason = f'exceeded {args.max_seconds}s deadline' if signum == signal.SIGALRM else 'interrupted'
            raise TimeoutError('benchmark ' + reason)
        previous_handlers = {sig: signal.signal(sig, expired)
                             for sig in (signal.SIGALRM, signal.SIGINT, signal.SIGTERM)}
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
        try:
            deadline = time.monotonic() + args.max_seconds
            while True:
                if process.poll() is not None:
                    raise RuntimeError('server exited: ' + out.with_suffix('.server.log').read_text()[-4000:])
                try:
                    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
                    conn.request('GET', '/health')
                    response = conn.getresponse()
                    ready = response.status == 200
                    response.read()
                    conn.close()
                    if ready:
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('server readiness timeout')
                time.sleep(.1)
            offset = 0
            for concurrency in args.concurrency:
                count = concurrency if args.single_wave else max(args.requests, concurrency * 2)
                level_jobs = jobs[offset:offset+count] if args.single_wave else jobs[:count]
                if args.single_wave: offset += count
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    list(executor.map(lambda job: request(port, job, warmup=True, options=options,
                                                         warmup_frames=args.warmup_frames), level_jobs[:concurrency]))
                    started = time.perf_counter()
                    barrier = threading.Barrier(concurrency) if args.simultaneous_start else None
                    runs = list(executor.map(lambda entry: request(port, entry[1], options=options,
                        run_start=started, arrival_at=started + entry[0] * args.arrival_spacing_ms / 1000
                        if entry[0] < concurrency else None,
                        launch_barrier=barrier if entry[0] < concurrency else None),
                        enumerate(level_jobs)))
                    elapsed = time.perf_counter() - started
                gaps = [b['at_ms'] - a['at_ms'] for run in runs
                        for a, b in zip(run['chunks'], run['chunks'][1:])]
                summary = dict(concurrency=concurrency, requests=len(runs), elapsed_s=elapsed,
                               qps=len(runs) / elapsed,
                               audio_seconds_per_second=sum(r['audio_seconds'] for r in runs) / elapsed)
                events = sorted((at, delta) for run in runs for at, delta in (
                    (run['started_at_ms'], 1), (run['started_at_ms'] + run['total_ms'], -1)))
                active = peak = 0
                for _, delta in events:
                    active += delta
                    peak = max(peak, active)
                summary['peak_inflight'] = peak
                initial = [run['started_at_ms'] for run in runs[:concurrency]]
                summary['initial_launch_span_ms'] = max(initial) - min(initial)
                for metric in ('ttft_ms', 'tpot_ms', 'total_ms', 'audio_seconds'):
                    values = [r[metric] for r in runs]
                    summary[metric] = dict(p50=statistics.median(values), p95=percentile(values, .95),
                                           p99=percentile(values, .99))
                summary['gap_ms'] = dict(p50=statistics.median(gaps), p95=percentile(gaps, .95),
                                         p99=percentile(gaps, .99)) if gaps else {}
                report['rows'].append(dict(summary=summary, requests=runs))
                out.write_text(json.dumps(report, indent=2) + '\n')
                print(json.dumps(summary), flush=True)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            for sig, handler in previous_handlers.items(): signal.signal(sig, handler)
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', required=True)
    parser.add_argument('--model', default='/workspace/Breeze-TTS-2')
    parser.add_argument('--mode', default='int8')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--runtime-memory-mb', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--varied', action='store_true', help='distinct Chinese texts and 10–100 second frame budgets')
    parser.add_argument('--workload', help='JSON list of text/max_frames jobs; overrides generated text')
    parser.add_argument('--request-options', help='JSON file with shared request fields (e.g. reference audio/language)')
    parser.add_argument('--warmup-frames', type=int, default=24)
    parser.add_argument('--arrival-spacing-ms', type=float, default=0,
                        help='spacing of initial client arrivals; later requests keep the clients busy')
    parser.add_argument('--simultaneous-start', action='store_true',
                        help='barrier-release all initial client threads together; later requests are closed-loop')
    parser.add_argument('--single-wave', action='store_true',
                        help='one completed request per client at each concurrency, using distinct fixture entries')
    parser.add_argument('--max-seconds', type=float, default=60,
                        help='hard deadline for a server benchmark run, including initialization (maximum 60)')
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--requests', type=int, default=32)
    parser.add_argument('--concurrency', type=lambda s: [int(x) for x in s.split(',')], default=[16,32])
    parser.add_argument('--text', default='你好，今天我们验证并发请求的流式语音生成性能。')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.simultaneous_start and args.arrival_spacing_ms:
        parser.error('--simultaneous-start and --arrival-spacing-ms are mutually exclusive')
    if args.max_seconds <= 0 or args.max_seconds > 60:
        parser.error('--max-seconds must be between 0 and 60')
    if args.requests < 1 or not args.concurrency or min(args.concurrency) < 1:
        parser.error('request count and concurrency must be positive')
    run(args)
