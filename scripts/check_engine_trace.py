"""Check an Nsight SQLite export of an ordinary native serving benchmark."""
import argparse
import bisect
import json
import sqlite3
from pathlib import Path


def check(path, require_two_streams=False):
    connection = sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True)
    begin = connection.execute("SELECT MIN(start) FROM NVTX_EVENTS WHERE text='tts.engine.dispatch'").fetchone()[0]
    end = connection.execute("SELECT MAX(end) FROM NVTX_EVENTS WHERE text='tts.engine.complete'").fetchone()[0]
    if begin is None or end is None:
        raise RuntimeError('trace contains no completed engine execution')
    counts = dict(connection.execute('''SELECT s.value, COUNT(*)
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
        WHERE r.start BETWEEN ? AND ? GROUP BY s.value''', (begin, end)))
    submission_threads = [row[0] for row in connection.execute('''SELECT DISTINCT r.globalTid
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id
        WHERE r.start BETWEEN ? AND ? AND
        (s.value LIKE 'cudaGraphLaunch%' OR s.value LIKE 'cudaLaunchKernel%' OR
         s.value LIKE 'cuGraphLaunch%' OR s.value LIKE 'cuLaunchKernel%')''', (begin, end))]
    plans = connection.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='tts.engine.plan' AND start BETWEEN ? AND ?",
                               (begin, end)).fetchall()
    commits = connection.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='tts.engine.complete' AND start BETWEEN ? AND ?",
                                 (begin, end)).fetchall()
    gpu = sorted(connection.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start BETWEEN ? AND ?',
                                    (begin, end)).fetchall()
                 + connection.execute('SELECT start,end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE WHERE start BETWEEN ? AND ?',
                                      (begin, end)).fetchall())
    starts, latest_end = [], []
    for start, stop in gpu:
        starts.append(start)
        latest_end.append(max(stop, latest_end[-1] if latest_end else stop))
    def find_overlaps(ranges):
        overlaps = []
        for start, stop in ranges:
            index = bisect.bisect_left(starts, stop) - 1
            while index >= 0 and latest_end[index] > start:
                first, last = gpu[index]
                if last > start:
                    overlaps.append(dict(cpu_start_ns=start, cpu_end_ns=stop,
                                         gpu_start_ns=first, gpu_end_ns=last,
                                         overlap_ns=min(stop, last) - max(start, first)))
                    break
                index -= 1
        return overlaps
    overlaps = find_overlaps(plans)
    commit_overlaps = find_overlaps(commits)
    forbidden = ('cudaStreamSynchronize', 'cudaDeviceSynchronize', 'cudaEventSynchronize',
                 'cuStreamSynchronize', 'cuCtxSynchronize', 'cuEventSynchronize',
                 'cudaMalloc', 'cudaFree', 'cudaStreamBeginCapture', 'cudaGraphInstantiate')
    unexpected = {name: count for name, count in counts.items() if name.startswith(forbidden)}
    report = dict(window_seconds=(end - begin) / 1e9, cuda_submission_threads=submission_threads,
                  api_counts=counts, cpu_plans=len(plans), plans_overlapping_gpu=len(overlaps),
                  examples=overlaps[:4], cpu_commits=len(commits),
                  commits_overlapping_gpu=len(commit_overlaps), commit_examples=commit_overlaps[:4],
                  unexpected_api_calls=unexpected)
    compute_streams = sorted({r[0] for table in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE')
                              for r in connection.execute(f'SELECT DISTINCT streamId FROM {table} WHERE start BETWEEN ? AND ?', (begin, end))})
    copy_streams = [r[0] for r in connection.execute('SELECT DISTINCT streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start BETWEEN ? AND ?', (begin, end))]
    report['compute_streams'] = compute_streams
    report['copy_streams'] = sorted(copy_streams)
    for name, kind in (('h2d_streams', 1), ('d2h_streams', 2)):
        report[name] = sorted(r[0] for r in connection.execute(
            'SELECT DISTINCT streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start BETWEEN ? AND ? AND copyKind=?',
            (begin, end, kind)))
    report['stream_wait_event_calls'] = sum(count for name, count in counts.items() if name.startswith('cudaStreamWaitEvent'))
    runtime = list(connection.execute('SELECT r.start,r.end,r.correlationId,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id WHERE r.start BETWEEN ? AND ? ORDER BY r.start', (begin, end)))
    calls = {row[2]: row for row in runtime}
    queries = [row[0] for row in runtime if row[3].startswith('cudaEventQuery')]
    device = sorted((a,b,corr) for table in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE')
                    for a,b,corr in connection.execute(f'SELECT start,end,correlationId FROM {table} WHERE start BETWEEN ? AND ?', (begin,end)))
    submitted_device = sorted((calls[corr][0],a,b) for a,b,corr in device if corr in calls)
    submitted_times = [row[0] for row in submitted_device]
    readbacks = sorted((calls[corr][0],a,b,size) for a,b,size,corr in connection.execute(
        'SELECT start,end,bytes,correlationId FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start BETWEEN ? AND ? AND copyKind=2', (begin,end)) if corr in calls)
    readback_starts = [row[0] for row in readbacks]
    successors = []
    for first,last in connection.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='tts.runner.submit_next' AND start BETWEEN ? AND ? ORDER BY start", (begin,end)):
        own = [(a,b) for _,a,b in submitted_device[bisect.bisect_left(submitted_times,first):bisect.bisect_right(submitted_times,last)]]
        previous = bisect.bisect_right(readback_starts,first)-1
        if not own or previous < 0: continue
        copied_at,a,b,size = readbacks[previous]
        query = bisect.bisect_left(queries,copied_at)
        successors.append(dict(submit_start_ns=first,submit_end_ns=last,gpu_start_ns=min(x[0] for x in own),
            gpu_end_ns=max(x[1] for x in own),predecessor_pcm_start_ns=a,predecessor_pcm_end_ns=b,
            predecessor_pcm_bytes=size,submitted_before_first_query=query<len(queries) and last<=queries[query],
            pcm_compute_overlap_ns=sum(max(0,min(y,b)-max(x,a)) for x,y in own)))
    report['submitted_successors'] = len(successors)
    report['submitted_before_first_query'] = sum(r['submitted_before_first_query'] for r in successors)
    report['successors_with_pcm_compute_overlap'] = sum(r['pcm_compute_overlap_ns'] > 0 for r in successors)
    report['successor_examples'] = successors[:4]
    batches = connection.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='tts.engine.batch_forward' AND start BETWEEN ? AND ? ORDER BY start", (begin, end)).fetchall()
    report['scheduled_batch_forwards'] = len(batches)
    def covered(ranges, first, last):
        merged = []
        for start, stop in sorted(ranges):
            start, stop = max(start, first), min(stop, last)
            if stop <= start:
                continue
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(stop, merged[-1][1])
            else:
                merged.append([start, stop])
        return sum(stop - start for start, stop in merged)
    first_gpu = gpu[0][0]
    report['leading_no_compute_ms'] = (first_gpu - begin) / 1e6
    report['compute_graph_span_coverage_percent'] = 100 * covered(gpu, begin, end) / (end - begin)
    report['coverage_after_first_compute_percent'] = 100 * covered(gpu, first_gpu, end) / (end - first_gpu)
    report['coverage_definition'] = 'Union of kernel and graph execution spans, not SM utilization. CPU IP sampling is disabled.'
    if connection.execute("SELECT 1 FROM sqlite_master WHERE name='SCHED_EVENTS'").fetchone():
        cpu = {}
        for tid in submission_threads:
            earlier = connection.execute('SELECT isSchedIn FROM SCHED_EVENTS WHERE globalTid=? AND start<=? ORDER BY start DESC LIMIT 1', (tid,begin)).fetchone()
            active = begin if earlier and earlier[0] else None
            running = 0
            for at,on in connection.execute('SELECT start,isSchedIn FROM SCHED_EVENTS WHERE globalTid=? AND start>? AND start<? ORDER BY start,isSchedIn', (tid,begin,end)):
                if on and active is None: active=at
                elif not on and active is not None: running+=at-active;active=None
            if active is not None: running+=end-active
            cpu[str(tid)] = dict(on_core_seconds=running/1e9,on_core_percent=100*running/(end-begin))
        report['submission_thread_cpu'] = cpu
        report['cpu_definition'] = 'Scheduled CPU runtime per submission thread, including polling; percentage of one core, not whole-machine utilization.'
    separate = (len(compute_streams) == len(report['d2h_streams']) == 1
                and not set(compute_streams).intersection(report['d2h_streams'])
                and set(report['h2d_streams']).issubset(compute_streams))
    if len(submission_threads) != 1 or unexpected or not overlaps or (require_two_streams and (not separate or not report['stream_wait_event_calls'])):
        raise RuntimeError(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sqlite')
    parser.add_argument('--output')
    parser.add_argument('--require-two-streams', action='store_true')
    args = parser.parse_args()
    report = json.dumps(check(args.sqlite, args.require_two_streams), indent=2) + '\n'
    if args.output:
        Path(args.output).write_text(report)
    print(report, end='')
