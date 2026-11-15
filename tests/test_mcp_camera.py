"""Camera tools share backend lifecycle; paired files publish only after both succeed."""
import asyncio,json,unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import httpx,numpy as np
from pydantic import ValidationError
from flowhmr.mcp.backend import Backend
from flowhmr.mcp.contract import ExportOptions
from flowhmr.mcp.exports import Exports
from flowhmr.mcp.tools import build_tools

def archive(files):
    output=BytesIO()
    with ZipFile(output,'w') as z:
        for name,value in files.items():z.writestr(name,value)
    return output.getvalue()

class CameraMCPTests(unittest.TestCase):
    def backend(self,handler):
        backend=Backend('http://backend')
        backend.client=lambda timeout=30:httpx.AsyncClient(base_url=backend.url,transport=httpx.MockTransport(handler))
        return backend

    def test_tools_prepare_status_cancel_and_retry_use_the_same_routes(self):
        calls=[];state='none'
        def handler(r):
            nonlocal state
            calls.append((r.method,r.url.path))
            if r.url.path.endswith('/activity'):return httpx.Response(200,json={'id':'receipt'})
            if r.url.path.endswith('/prepare'):state='queued'
            if r.url.path.endswith('/cancel'):state='cancel_requested'
            return httpx.Response(200,json={'camera_state':state})
        tools={t.name:t for t in build_tools(self.backend(handler),None)}
        async def run():
            for name in ['flowhmr_prepare_camera','flowhmr_get_camera_status','flowhmr_cancel_camera','flowhmr_prepare_camera']:
                result=await tools[name].run({'job_id':'sample'})
                self.assertIn('camera_state',result.model_dump())
        asyncio.run(run())
        self.assertEqual([p.rsplit('/',1)[-1] for _,p in calls if '/camera/' in p],['prepare','status','cancel','prepare'])

    def test_paired_npz_keeps_npz_intact_and_camera_failure_publishes_nothing(self):
        npz=BytesIO();np.savez(npz,trans=np.zeros((2,3)));motion=npz.getvalue()
        camera=archive({'camera.npz':motion,'camera.json':'{}'})
        for failure in [False,True]:
            with self.subTest(failure=failure),TemporaryDirectory() as tmp:
                root=Path(tmp);target=root/'paired.zip';calls=[]
                def handler(r):
                    calls.append(r.url.path)
                    if r.url.path.endswith('/prepare'):return httpx.Response(200,json={'camera_state':'ready'})
                    if r.url.path.endswith('/camera/export'):
                        return httpx.Response(500,json={'error':'camera failed'}) if failure else httpx.Response(200,content=camera)
                    if r.url.path.endswith('/npz'):return httpx.Response(200,content=motion)
                    return httpx.Response(200,json={'id':'sample','status':'done','selected_person':0})
                exports=Exports(root/'receipts',self.backend(handler))
                async def run():
                    row=await exports.submit(ExportOptions(job_id='sample',format='npz',include_camera=True,output_path=str(target)))
                    await exports.close();return exports.get(row['export_id'])
                result=asyncio.run(run());self.assertEqual(result['status'],'failed' if failure else 'done')
                self.assertEqual(target.exists(),not failure)
                if not failure:
                    with ZipFile(target) as z:
                        self.assertEqual(z.read('motion/motion.npz'),motion)
                        self.assertIn('camera/camera.npz',z.namelist())
                self.assertFalse(list(root.glob('.flowhmr-*')))
                self.assertNotIn('/api/upload',calls)

    def test_gap_zip_never_masquerades_as_a_vmd(self):
        payload=archive({'motion.vmd':b'vmd','motion-report.json':b'{}'})
        for suffix in ['vmd','zip']:
            with TemporaryDirectory() as tmp:
                root=Path(tmp);target=root/('motion.'+suffix)
                def handler(r):
                    if r.url.path.endswith('/vmd'):return httpx.Response(200,content=payload)
                    return httpx.Response(200,json={'id':'sample','status':'done','selected_person':0})
                exports=Exports(root/'receipts',self.backend(handler))
                async def run():
                    row=await exports.submit(ExportOptions(job_id='sample',format='vmd',output_path=str(target)))
                    await exports.close();return exports.get(row['export_id'])
                row=asyncio.run(run())
                if suffix=='vmd':
                    self.assertEqual(row['error']['code'],'export_requires_zip');self.assertFalse(target.exists())
                else:
                    with ZipFile(target) as z:self.assertEqual(z.read('motion.vmd'),b'vmd')

    def test_old_options_stay_valid_and_unsupported_camera_formats_are_rejected(self):
        self.assertFalse(ExportOptions(job_id='sample',format='npz',output_path='unused').include_camera)
        for options in [dict(format='vrma'),dict(format='fbx',fbx_profile='mixamo')]:
            with self.assertRaises(ValidationError):ExportOptions(job_id='sample',output_path='unused',include_camera=True,**options)

if __name__=='__main__':unittest.main()
