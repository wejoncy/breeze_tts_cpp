"""Draw four actual scheduled batches from an Nsight SQLite export (no GPU work)."""
import argparse
import bisect
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def draw(path, output):
    c = sqlite3.connect(f'file:{Path(path).resolve()}?mode=ro', uri=True)
    origin = c.execute("SELECT MIN(start) FROM NVTX_EVENTS WHERE text='tts.engine.dispatch'").fetchone()[0]
    forwards = c.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='tts.engine.batch_forward' ORDER BY start").fetchall()
    commits = c.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='tts.engine.complete' ORDER BY start").fetchall()
    # A bounded capture can end with one batch still in flight.
    completed = min(len(forwards), len(commits))
    forwards, commits = forwards[:completed], commits[:completed]
    if completed < 4:
        raise RuntimeError('expected at least four completed generation batches')
    starts = [a for a, _ in forwards]
    successors = c.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='tts.runner.submit_next' ORDER BY start").fetchall()
    successor_starts = [a for a, _ in successors]
    groups = [[] for _ in forwards]
    # CUPTI correlation links device activities to their actual host submissions.
    # Early submit_next activities belong to the following adopted batch.
    # Graph spans already cover graph kernels.
    for table in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE'):
        for a, b, submitted in c.execute(f'''SELECT g.start,g.end,r.start FROM {table} g
                JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON g.correlationId=r.correlationId
                WHERE r.start>=? ORDER BY g.start''', (starts[0],)):
            index = bisect.bisect_right(starts, submitted) - 1
            successor = bisect.bisect_right(successor_starts, submitted) - 1
            if successor >= 0 and submitted <= successors[successor][1]:
                index = bisect.bisect_right(starts, successors[successor][1])
            if index < len(groups):
                groups[index].append((a, b))
    ms = lambda ns: (ns - origin) / 1e6
    # Prefer four steady full-frame batches; first batches contain prefills.
    candidates = [i for i in range(len(forwards) - 3) if all(groups[j] for j in range(i, i + 4))]
    if not candidates:
        raise RuntimeError('no four consecutive correlated batches')
    steady = [i for i in candidates if i >= 8 and all(len(groups[j]) == 2 for j in range(i,i+4))]
    start_index = steady[0] if steady else next((i for i in candidates if i >= 8), candidates[0])
    selected = range(start_index, start_index + 4)
    records = []
    for index in selected:
        rows = groups[index]
        if not rows:
            raise RuntimeError('batch has no correlated GPU activity')
        records.append(dict(batch=index + 1, forward_start_ms=ms(forwards[index][0]),
                            forward_end_ms=ms(forwards[index][1]),
                            gpu_first_ms=ms(min(a for a, _ in rows)),
                            gpu_last_ms=ms(max(b for _, b in rows)),
                            commit_start_ms=ms(commits[index][0]),
                            commit_end_ms=ms(commits[index][1])))
    low = min(starts[start_index], min(a for index in selected for a,_ in groups[index])) - 100000
    high = commits[start_index+3][1]
    copies = c.execute('SELECT start,end,copyKind,streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND end<=? ORDER BY start',
                       (low, high)).fetchall()
    compute_streams = sorted({row[0] for table in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_GRAPH_TRACE')
                              for row in c.execute(f'SELECT DISTINCT streamId FROM {table} WHERE start>=? AND end<=?',
                                                   (low,high))})
    copy_streams = sorted({s for _,_,k,s in copies if k == 2})
    cpu = {name: c.execute('SELECT start,end FROM NVTX_EVENTS WHERE text=? AND start>=? AND start<=? ORDER BY start',
                           (name, low, high)).fetchall()
           for name in ('tts.engine.plan', 'tts.engine.dispatch', 'tts.runner.submit_next')}
    colors = ['#227b67', '#327bb2', '#227b67', '#327bb2']
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), layout='constrained', height_ratios=[1.6, 1])
    for ax in axes:
        for index in selected:
            intervals = [(ms(a), (b - a) / 1e6) for a, b in groups[index]]
            ax.broken_barh(intervals, (1.7, .6), facecolors=colors[index % 4], edgecolors='none')
            a, b = forwards[index]
            ax.broken_barh([(ms(a), (b-a)/1e6)], (2.7,.6), facecolors='#7284a9')
            a, b = commits[index]
            ax.broken_barh([(ms(a), (b-a)/1e6)], (-.3,.6), facecolors='#d87925')
        for name, y, color in [('tts.engine.plan', 5, '#5678cd'), ('tts.engine.dispatch', 4, '#a35bb3'), ('tts.runner.submit_next', 3, '#245cc1')]:
            for a, b in cpu[name]:
                ax.broken_barh([(ms(a),(b-a)/1e6)], (y-.3,.6), facecolors=color)
                if ax is axes[0]:
                    ax.vlines(ms(a),y-.3,y+.3,color=color,linewidth=1)
        for kind, color in [(1,'#c14761'),(8,'#a13e60'),(2,'#d7a026')]:
            rows = [(ms(a),(b-a)/1e6) for a,b,k,_ in copies if k==kind]
            ax.broken_barh(rows,((1.7 if kind in (1,8) else .7),.6),facecolors=color,edgecolors='none')
            if ax is axes[0]:
                ax.vlines([a for a,_ in rows],.75,1.25,color=color,linewidth=.7,alpha=.7)
        ax.set_yticks([5,4,3,2,1,0],['CPU prepare','CPU schedule','CPU submit next / adopt current',
                                    f'GPU compute (stream {compute_streams})',
                                    f'GPU D2H (stream {copy_streams})','CPU output commit'])
        ax.set_ylim(-.6,5.6)
        ax.grid(axis='x',alpha=.2)
        ax.set_axisbelow(True)
        for side in ['top','right','left']:
            ax.spines[side].set_visible(False)
        ax.set_xlabel('Milliseconds from first Engine dispatch (actual Nsight timestamps)')
    first, last = records[0]['forward_start_ms']-1, records[3]['commit_end_ms']+1
    axes[0].set_xlim(first,last)
    for record in records:
        axes[0].text((record['gpu_first_ms']+record['gpu_last_ms'])/2,2,f"Batch {record['batch']}",
                     ha='center',va='center',color='white',fontsize=10)
        axes[0].annotate(f"Commit {record['batch']}",(record['commit_start_ms'],0),
                         (record['commit_start_ms']-1,-.55),ha='right',fontsize=8)
    overlapping_plans = [(a,b) for a,b in cpu['tts.engine.plan']
                         if records[0]['gpu_first_ms'] < ms(a) < records[0]['gpu_last_ms']]
    if overlapping_plans:
        plan = overlapping_plans[0]
        axes[0].annotate('Prepare future inputs while Batch 1 runs', (ms(plan[0]),5),
                         (ms(plan[0])+4,5.45),fontsize=9,arrowprops=dict(arrowstyle='->',color='#5678cd'))
    zoom = records[2]
    axes[1].set_xlim(min(zoom['gpu_first_ms'],records[1]['commit_start_ms'])-.06,
                     max(zoom['gpu_first_ms'],records[1]['commit_end_ms'])+.18)
    axes[1].set_title(f"Zoom: Batch {records[2]['batch']} and Commit {records[1]['batch']}",loc='left',fontsize=11)
    axes[1].annotate(f"Commit {records[1]['batch']}; GPU batch {records[2]['batch']}", (records[1]['commit_start_ms'],0),
                     (records[1]['commit_start_ms']-.17,.35),fontsize=9,
                     arrowprops=dict(arrowstyle='->',color='#d87925'))
    axes[0].set_title('Four scheduled batches: one Engine thread, two event-ordered CUDA streams',
                       loc='left',fontweight='bold',fontsize=13)
    fig.legend(handles=[Patch(color='#c14761',label='H2D'),Patch(color='#d7a026',label='D2H'),
                        Patch(color='#d87925',label='CPU commit'),Patch(color='#245cc1',label='CPU submit next')],loc='outside lower center',ncol=3)
    fig.suptitle('GPU: measured kernel/graph spans. CPU: wall intervals, not utilization. Narrow overview marks show positions.',fontsize=9)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):
        fig.savefig(output.with_suffix('.'+suffix),dpi=180)
    evidence = dict(assignment='GPU correlationId -> host runtime; submit_next scope belongs to the following adopted batch',
                    records=records, cpu_ranges_ms={k:[(ms(a),ms(b)) for a,b in v] for k,v in cpu.items()},
                    copy_streams=sorted({s for _,_,_,s in copies}),
                    note='Graph execution envelopes are not SM occupancy. CPU sampling is disabled.')
    output.with_suffix('.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite')
    parser.add_argument('--output',required=True,help='Output file stem for PNG/SVG/JSON')
    args = parser.parse_args()
    draw(args.sqlite,args.output)
