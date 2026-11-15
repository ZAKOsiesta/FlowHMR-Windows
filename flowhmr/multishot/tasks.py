"""Queue-independent camera task service, shared by HTTP and future MCP callers."""
from pathlib import Path
from threading import RLock
import time
from zipfile import BadZipFile

from .camera import Camera
from .recovery import CameraCancelled
from .runtime import camera_directory
from .quality import read_quality, ready_message

ACTIVE_CAMERA_STATES = frozenset(('queued', 'running', 'cancel_requested'))


class CameraTasks:
    def __init__(self, get_job, job_directory, update_job, enqueue, build):
        self.get_job, self.directory, self.update = get_job, job_directory, update_job
        self.enqueue, self.build = enqueue, build
        self.lock = RLock()

    def root(self, job_id):
        job = self.get_job(job_id)
        if job['status'] != 'done':
            raise ValueError('请等待人物动作完成，再恢复摄像机。')
        if job.get('person_batch') == job_id and job.get('scene_state') != 'ready':
            raise ValueError('请先完成多人场景对齐，再恢复共同机位。')
        return camera_directory(self.directory(job_id), job)

    def ready(self, job_id):
        try:
            Camera.load(self.root(job_id) / 'camera/camera.npz')
            return True
        except (OSError, ValueError, KeyError, BadZipFile):
            return False

    def status(self, job_id):
        job = self.get_job(job_id)
        state = job.get('camera_state', 'none')
        if state not in ACTIVE_CAMERA_STATES:
            if self.ready(job_id):
                state = 'ready'
            elif state == 'ready':
                state = 'none'
        quality = read_quality(self.root(job_id)) if state == 'ready' else None
        return dict(camera_state=state, message=ready_message(quality) if quality else job.get('camera_message', ''),
                    quality=quality,
                    progress=job.get('camera_progress', 0), error=job.get('camera_error'),
                    started=job.get('camera_started'), finished=job.get('camera_finished'))

    def prepare(self, job_id):
        with self.lock:
            self.root(job_id)
            status = self.status(job_id)
            if status['camera_state'] in ACTIVE_CAMERA_STATES or status['camera_state'] == 'ready':
                return status
            self.update(job_id, camera_state='queued', camera_message='等待恢复摄像机', camera_progress=0,
                        camera_error=None, camera_quality=None, camera_started=None, camera_finished=None)
            self.enqueue(('camera', job_id))
            return self.status(job_id)

    def cancel(self, job_id):
        with self.lock:
            if self.get_job(job_id).get('camera_state') in ACTIVE_CAMERA_STATES:
                self.update(job_id, camera_state='cancel_requested', camera_message='正在停止摄像机计算')
            return self.status(job_id)

    def run(self, job_id):
        def check():
            if self.get_job(job_id).get('camera_state') == 'cancel_requested':
                raise CameraCancelled()

        def notify(message, progress):
            check()
            self.update(job_id, camera_message=message, camera_progress=float(progress))

        try:
            with self.lock:
                check()
                self.root(job_id)
                self.update(job_id, camera_state='running', camera_started=time.time(), camera_error=None)
            self.build(job_id, check=check, notify=notify)
            # A complete, atomically committed result wins over a late cancel.
            if not self.ready(job_id):
                raise ValueError('摄像机结果未完整保存。')
            quality = read_quality(self.root(job_id))
            self.update(job_id, camera_state='ready', camera_progress=100,
                        camera_quality=quality, camera_message=ready_message(quality), camera_finished=time.time())
        except CameraCancelled:
            self.update(job_id, camera_state='cancelled', camera_message='摄像机计算已停止，人物动作已保留',
                        camera_error=None, camera_finished=time.time())
        except Exception as exc:
            self.update(job_id, camera_state='failed', camera_message='摄像机恢复失败，可重试',
                        camera_error=f'{type(exc).__name__}: {exc}', camera_finished=time.time())


def restore_camera_state(job):
    if job.get('camera_state') in ACTIVE_CAMERA_STATES:
        job.update(camera_state='failed', camera_error='应用重启中断了相机计算，可重试；人物动作已保留。',
                   camera_message='摄像机计算已中断', camera_finished=time.time())
