"""Optional post-task camera endpoints; no model generation or job status mutation."""
from pathlib import Path
from threading import Lock

from flask import jsonify, request, send_file

from .camera import Camera
from .exports import camera_bundle


def register_camera_routes(app, get_job, job_directory, load_joints, tasks=None):
    export_lock = Lock()

    def root(job_id):
        if tasks is not None:
            return tasks.root(job_id)
        job = get_job(job_id)
        if job['status'] != 'done':
            raise ValueError('人物动作尚未完成。')
        base = Path(job_directory(job_id))
        return base / 'scene' if job.get('person_batch') == job_id else base

    def missing():
        return jsonify(error='此任务还没有可用摄像机。', error_code='camera_not_ready'), 409

    @app.get('/api/jobs/<job_id>/camera')
    def paired_camera(job_id):
        try:
            from .ground import stored_floor
            directory=root(job_id)
            return jsonify(Camera.load(directory / 'camera/camera.npz').json(stored_floor(directory)))
        except FileNotFoundError:
            return missing()
        except (ValueError, KeyError) as exc:
            return jsonify(error=str(exc), error_code='camera_invalid'), 422

    @app.post('/api/jobs/<job_id>/camera/prepare')
    def prepare_paired_camera(job_id):
        try:
            if tasks is not None:
                status = tasks.prepare(job_id)
                return jsonify(status), 200 if status['camera_state'] == 'ready' else 202
            Camera.load(root(job_id) / 'camera/camera.npz')
            return jsonify(camera_state='ready')
        except FileNotFoundError:
            return missing()
        except (ValueError, KeyError) as exc:
            return jsonify(error=str(exc), error_code='camera_invalid'), 422

    @app.get('/api/jobs/<job_id>/camera/status')
    def paired_camera_status(job_id):
        return jsonify(tasks.status(job_id) if tasks else {'camera_state': 'ready'})

    @app.post('/api/jobs/<job_id>/camera/cancel')
    def cancel_paired_camera(job_id):
        return jsonify(tasks.cancel(job_id) if tasks else {'camera_state': 'ready'})

    @app.get('/api/jobs/<job_id>/camera/report')
    def paired_camera_report(job_id):
        path = root(job_id) / 'camera/report.json'
        if not path.is_file():
            return missing()
        return send_file(path, mimetype='application/json')

    @app.post('/api/jobs/<job_id>/camera/export')
    def export_paired_camera(job_id):
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or set(body) - {'format', 'transform'}:
            return jsonify(error='摄像机导出参数无效。', error_code='camera_options'), 400
        try:
            with export_lock:
                directory = root(job_id)
                if not (directory / 'camera/camera.npz').is_file():
                    return missing()
                kind = body.get('format')
                joints = load_joints(job_id, directory) if kind == 'vmd' else None
                stream = camera_bundle(directory, kind, joints, body.get('transform'))
            return send_file(stream, mimetype='application/zip', as_attachment=True, download_name='camera.zip')
        except (ValueError, KeyError, TypeError) as exc:
            return jsonify(error=str(exc), error_code='camera_options'), 400
