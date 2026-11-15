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

    def test_ready_status_retains_gravity_degradation_without_expanded_report(self):
        quality = dict(degraded=True, reason='weights_missing', fallback_shots=[1], total_shots=1)
        def handler(r):
            if r.url.path.endswith('/activity'):
                return httpx.Response(200, json={'id':'receipt'})
            return httpx.Response(200, json=dict(camera_state='ready', quality=quality))
        tools = {t.name:t for t in build_tools(self.backend(handler), None)}
        result = asyncio.run(tools['flowhmr_get_camera_status'].run({'job_id':'sample'}))
        self.assertEqual(result.model_dump()['quality'], quality)

    def test_gap_zip_retry_works_without_masquerading_as_a_motion_file(self):
        for format_name in ['vmd','fbx','vrma']:
          payload=archive({f'motion.{format_name}':format_name.encode(),'motion-report.json':b'{}'})
          for suffix in [format_name,'zip']:
            with self.subTest(format=format_name,suffix=suffix),TemporaryDirectory() as tmp:
                root=Path(tmp);target=root/('motion.'+suffix)
                def handler(r):
                    if r.url.path.endswith('/'+format_name):return httpx.Response(200,content=payload)
                    return httpx.Response(200,json={'id':'sample','status':'done','selected_person':0,'has_fbx':True})
                exports=Exports(root/'receipts',self.backend(handler))
                async def run():
                    row=await exports.submit(ExportOptions(job_id='sample',format=format_name,output_path=str(target)))
                    await exports.close();return exports.get(row['export_id'])
                row=asyncio.run(run())
                if suffix==format_name:
                    self.assertEqual(row['error']['code'],'export_requires_zip');self.assertFalse(target.exists())
                else:
                    self.assertEqual(row['status'],'done')
                    with ZipFile(target) as z:
                        self.assertEqual(z.read(f'motion.{format_name}'),format_name.encode())
                        self.assertEqual(z.read('motion-report.json'),b'{}')

    def test_old_options_stay_valid_and_unsupported_camera_formats_are_rejected(self):
        self.assertFalse(ExportOptions(job_id='sample',format='npz',output_path='unused').include_camera)
        for options in [dict(format='vrma'),dict(format='fbx',fbx_profile='mixamo')]:
            with self.assertRaises(ValidationError):ExportOptions(job_id='sample',output_path='unused',include_camera=True,**options)

    def test_plain_vrma_multi_vrma_and_native_npz_keep_their_containers(self):
        npz=BytesIO();np.savez(npz,trans=np.zeros((2,3)))
        multiple=archive({'person_01.vrma':b'glTF-first','person_02.vrma':b'glTF-second'})
        for fmt,suffix,people,payload in [('vrma','vrma',1,b'glTF-single'),
                                        ('vrma','zip',1,b'glTF-single'),
                                        ('vrma','zip',2,multiple),('npz','npz',1,npz.getvalue())]:
            with self.subTest(format=fmt,suffix=suffix,people=people),TemporaryDirectory() as tmp:
                root=Path(tmp);target=root/('motion.'+suffix)
                def handler(r):
                    if r.url.path.endswith('/'+fmt):return httpx.Response(200,content=payload)
                    job={'id':'sample','status':'done','selected_person':0}
                    if people==2:job['scene']={'people':[{'person':1},{'person':2}]}
                    return httpx.Response(200,json=job)
                exports=Exports(root/'receipts',self.backend(handler))
                async def run():
                    row=await exports.submit(ExportOptions(job_id='sample',format=fmt,output_path=str(target)))
                    await exports.close();return exports.get(row['export_id'])
                row=asyncio.run(run());self.assertEqual(row['status'],'done')
                if suffix!='zip':self.assertEqual(target.read_bytes(),payload)
                else:
                    with ZipFile(target) as actual:
                        if people==1:self.assertEqual(actual.read('motion.vrma'),payload)
                        else:
                            with ZipFile(BytesIO(payload)) as expected:
                                self.assertEqual(actual.namelist(),expected.namelist())
                                for name in expected.namelist():self.assertEqual(actual.read(name),expected.read(name))

if __name__=='__main__':unittest.main()
