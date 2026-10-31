"""Portable offline reports for policy loops, native replay and action branches."""
import html
import json
from pathlib import Path
import shutil

from evaluation.planning.registry import project_path
from evaluation.reporting.support import digest


def _json(path):
    return json.loads(Path(path).read_text())


def _cases(root, source):
    report = _json(source)
    if report.get('synthetic') is True and isinstance(report.get('job'), dict):
        return [{**report['job'], 'id': source.parent.name,
                 'status': report['status'], 'synthetic': True,
                 'directory': source.parent}], []
    if 'source_plan_sha256' in report and isinstance(report.get('jobs'), list):
        progress_path = source.parent / 'progress.json'
        progress = _json(progress_path) if progress_path.exists() else {}
        if progress and progress.get('study_sha256') != digest(source):
            raise ValueError('Native study progress is bound to a different frozen study')
        jobs = progress.get('jobs', {})
        cases = []
        for job in report['jobs']:
            record = jobs.get(job['id'], {})
            directory = source.parent / 'runs' / job['id']
            directory = project_path(root, str(directory))
            cases.append({**record, 'id': job['id'], 'status': record.get('status', 'not_run'),
                          'directory': directory if directory.is_dir() else None})
        return cases, []
    if report.get('kind') == 'action_branch_evaluation':
        return [{**j, 'directory': project_path(root, j['summary']).parent if j.get('summary') else None}
                for j in report['jobs']], report.get('branch_contrasts', [])
    if 'execution' in report or 'preflight' in report:
        jobs = report.get('execution', {}).get('jobs', [])
        planned = report.get('preflight', {}).get('selection', jobs)
        def key(j):
            return (j.get('id'), j.get('episode', j.get('episode_index')), j.get('seed'))
        indexed = {key(j): j for j in jobs}
        cases = []
        for item in planned:
            job = indexed.get(key(item), {**item, 'status': 'not_run'})
            cases.append({**job, 'directory': project_path(root, job['output']) if job.get('output') else None})
        return cases, []
    if 'world_model_calls' in report and 'files' in report:
        return [{'id': source.parent.name, 'status': report['status'], 'directory': source.parent}], []
    raise ValueError('Expected an evaluate report, native summary, demo summary or action-branch summary')


def build_dashboard(root, sources, output):
    root = Path(root).resolve()
    sources = [project_path(root, str(s)) for s in sources]
    output = project_path(root, str(output))
    if not sources or len(set(sources)) != len(sources):
        raise ValueError('Distinct nonempty report sources required')
    # Parse sources before creating output. Do not silently skip unsupported plans.
    parsed = [(p, *_cases(root, p)) for p in sources]
    output.mkdir(parents=True, exist_ok=False)
    (output / 'media').mkdir()
    report = {'schema_version': 1, 'kind': 'offline_evaluation_dashboard', 'cases': [],
              'sources': [{'path': str(p.relative_to(root)), 'sha256': digest(p)} for p in sources],
              'branch_contrasts': [], 'physical_reliability_verified': False,
              'scope': 'Provided planned slots only; heterogeneous domains/protocols are not a leaderboard.'}
    for source, cases, contrasts in parsed:
        report['branch_contrasts'].extend(contrasts)
        for case in cases:
            row = {'id': str(case.get('id', 'case')), 'status': case.get('status', 'unknown'),
                   'seed': case.get('seed'), 'family': case.get('family'), 'videos': [],
                   'metrics': {}, 'reference_available': False, 'cost': {}, 'error': case.get('error'),
                   'source': str(source.relative_to(root))}
            if case.get('synthetic'):
                row.update(synthetic=True, task_success_evaluated=False, model_weights_used=False)
            report['cases'].append(row)
            directory = case['directory']
            if directory is None:
                row['error'] = row['error'] or 'No execution artifact for this planned slot'
                continue
            try:
                summary_path = directory / 'summary.json'
                native = _json(summary_path) if summary_path.exists() else {}
                if 'files' in native and 'world_model_calls' in native:
                    row['status'] = native['status']
                    videos = {name: name for name in native['files'] if name.endswith('.mp4')}
                    for name, expected in native['files'].items():
                        path = _artifact(directory, name)
                        if digest(path) != expected:
                            raise ValueError(f'Artifact checksum mismatch: {name}')
                    evaluation = native.get('reference_evaluation', {})
                    row['reference_available'] = bool(evaluation)
                    row['metrics'] = evaluation.get('metrics', {})
                    row['metric_domain'] = evaluation.get('prediction_domain')
                    row['cost'] = {'generation_seconds': sum(native.get('timings', [])),
                                   'model_load_seconds': native.get('runtime', {}).get('load_seconds'),
                                   'memory_peak': native.get('worker_memory_peak')}
                    row['error'] = native.get('error', row['error'])
                    row['fps'] = native.get('video_fps')
                    row['frames'] = native.get('frames_per_view')
                    row['protocol_sha256'] = native.get('config_sha256')
                    row['world_model_calls'] = native['world_model_calls']
                else:
                    if row['status'] == 'completed':
                        from evaluation.reporting.artifact_checks import audit_job
                        audit = audit_job({**case, 'output': str(directory)})
                        if not audit['ok']:
                            raise ValueError('; '.join(audit['failures']))
                    meta = _json(directory / 'metadata.json')
                    videos = dict(meta['view_files'])
                    row['fps'], row['frames'] = meta['video_fps'], meta['num_video_frames']
                    row['pair_id'] = meta['pair_manifest']['pair_id']
                    row['state_sources'] = sorted(set(meta.get('state_sources', [])))
                    row['cost'] = {'generation_seconds': case.get('seconds'), 'model_load_seconds': None}
                    row['vla_calls'] = case.get('vla_calls')
                    row['world_model_calls'] = case.get('world_model_calls')
                    if not (directory / 'actions.npz').is_file():
                        raise ValueError('Policy action artifact missing')
                for label, relative in videos.items():
                    path = _artifact(directory, relative)
                    filename = f'{len(report["cases"]):04d}_{len(row["videos"]):02d}.mp4'
                    shutil.copyfile(path, output / 'media' / filename)
                    row['videos'].append({'label': label, 'file': 'media/' + filename, 'sha256': digest(path),
                                          'kind': 'prediction_vs_reference' if '_prediction_vs_reference' in relative else 'prediction'})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row.update(status='artifact_error', error=str(exc), videos=[], metrics={}, reference_available=False)
    cases = report['cases']
    report['counts'] = {'planned': len(cases), 'completed': sum(c['status'] == 'completed' for c in cases),
                        'failed': sum(c['status'] in {'failed', 'artifact_error'} for c in cases),
                        'other_or_not_run': sum(c['status'] not in {'completed', 'failed', 'artifact_error'} for c in cases),
                        'with_reference_metrics': sum(bool(c['metrics']) for c in cases),
                        'reference_unknown': sum(not c['reference_available'] for c in cases)}
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (output / 'index.html').write_text(render_html(report))
    return {'status': 'built', 'directory': str(output), 'counts': report['counts'], 'execution_performed': False}


def _artifact(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f'Missing or escaping artifact: {relative}')
    return path


def render_html(report):
    sections = []
    for case in report['cases']:
        e = html.escape
        videos = ''.join(f'<figure><figcaption>{e(v["label"])}</figcaption><video controls preload="metadata" src="{e(v["file"], quote=True)}"></video></figure>' for v in case['videos'])
        metrics = []
        for view, values in case['metrics'].items():
            series = values.get('per_frame', {}).get('mse_01', [])
            if series:
                ceiling = max(max(series), 1e-12)
                points = ' '.join(f'{i * 600 / max(1,len(series)-1):.2f},{110-v/ceiling*100:.2f}' for i,v in enumerate(series))
                metrics.append(f'<p>{e(view)} · Future-frame MSE (0–1), max {ceiling:.5g}</p><svg viewBox="0 0 600 120" role="img" aria-label="Future-frame error"><polyline fill="none" stroke="#5ec4e8" stroke-width="2" points="{points}"/></svg>')
        detail = {k: v for k, v in case.items() if k not in {'videos', 'metrics'}}
        notice = ('<p>Synthetic engineering demo · no pretrained model or task-success evaluation.</p>'
                  if case.get('synthetic') else '')
        sections.append(f'<section><h2>{e(case["id"])} <small>{e(case["status"])}</small></h2>'
                        f'{notice}'
                        f'<button class="play">Play views together</button> <button class="pause">Pause</button> '
                        '<label>Timeline <input class="seek" type="range" min="0" max="0" step="0.01" value="0"></label>'
                        f'<div class="views">{videos or "No video available"}</div>{"".join(metrics)}'
                        f'<details><summary>Reference metrics</summary><pre>{e(json.dumps(case["metrics"],indent=2,ensure_ascii=False))}</pre></details>'
                        f'<details><summary>Protocol, cost and errors</summary><pre>{e(json.dumps(detail,indent=2,ensure_ascii=False))}</pre></details></section>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>VLAWorldEval · Evaluation report</title><style>
body{font:16px system-ui;background:#111824;color:#e4edf6;max-width:1300px;margin:auto;padding:30px}h1{color:#67d5e7}small{font-size:14px;color:#abc}section{background:#1b2737;padding:22px;margin:20px 0;border-radius:14px}.views{display:flex;gap:16px;flex-wrap:wrap}figure{flex:1;min-width:260px;margin:14px 0}video{width:100%;max-height:430px}button{padding:9px;border:0;border-radius:7px;cursor:pointer}pre{white-space:pre-wrap;overflow-wrap:anywhere}svg{max-height:150px;width:100%}input{max-width:280px}a{color:#67d5e7}
</style><h1>VLAWorldEval</h1><p>Offline report · synchronized views · full provided denominator</p>
<p>Image similarity and action response are not physical reliability. Comparison videos: prediction left, recorded future right. Metrics exclude the initial frame. No aggregate ranking across protocols.</p>
<pre>''' + html.escape(json.dumps(report['counts'], indent=2)) + '''</pre><a href="report.json">Machine-readable report</a>''' + ''.join(sections) + '''<script>
for(const section of document.querySelectorAll('section')){
 const videos=[...section.querySelectorAll('video')], seek=section.querySelector('.seek');
 function duration(){const ds=videos.map(v=>v.duration).filter(Number.isFinite);seek.max=ds.length?Math.min(...ds):0;}
 videos.forEach(v=>v.addEventListener('loadedmetadata',duration));
 section.querySelector('.play').onclick=()=>{const t=Number(seek.value);videos.forEach(v=>{v.currentTime=t;v.play().catch(()=>{});});};
 section.querySelector('.pause').onclick=()=>videos.forEach(v=>v.pause());
 seek.oninput=()=>videos.forEach(v=>{v.pause();v.currentTime=Number(seek.value);});
 if(videos[0])videos[0].addEventListener('timeupdate',()=>{seek.value=videos[0].currentTime;for(const v of videos.slice(1))if(!v.paused&&Math.abs(v.currentTime-videos[0].currentTime)>.15)v.currentTime=videos[0].currentTime;});
}
</script></html>'''
