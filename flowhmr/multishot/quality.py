"""Persistent image-gravity quality, including reports from older releases."""
import json
from pathlib import Path


def camera_quality(report):
    if report.get('shared_world'):
        return dict(status='not_applicable', degraded=False, reason='shared_world')
    rows = [row for row in report.get('shots', []) if row.get('gravity_source')]
    fallback = sorted({row['shot'] for row in rows if row['gravity_source'] != 'image'})
    total = len({row['shot'] for row in rows})
    diagnostic = report.get('gravity', {})
    if not total:
        return dict(status='unknown', degraded=True, reason='unverified', fallback_shots=[], total_shots=0)
    reason = diagnostic.get('reason')
    if fallback and not reason:
        reason = 'unstable' if diagnostic.get('status') in ('measured', 'archived') else 'legacy_fallback'
    return dict(status=('partial' if len(fallback) < total else 'fallback') if fallback else 'image',
                degraded=bool(fallback), reason=reason if fallback else None,
                fallback_shots=fallback, total_shots=total)


def read_quality(directory):
    try:
        report = json.loads((Path(directory) / 'camera/report.json').read_text(encoding='utf8'))
        return camera_quality(report)
    except (OSError, ValueError, TypeError, KeyError):
        return dict(status='unknown', degraded=True, reason='unverified', fallback_shots=[], total_shots=0)


def ready_message(quality):
    return '摄像机已就绪（已降级，地面倾角可能不准确）' if quality['degraded'] else '摄像机已就绪'
