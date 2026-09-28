import argparse, copy, json, shutil, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import studio
from src.providers import complete, ProviderError

class ProviderTests(unittest.TestCase):
    def config(self):return {'profiles':{'a':{'model':'one'},'b':{'model':'two'}},'routes':{'writer':['a','b']},'retries_per_profile':1}
    def test_quota_retries_then_fallback(self):
        seen=[]
        def call(p,m):
            seen.append(p['model'])
            if p['model']=='one':raise ProviderError('HTTP 429',True)
            return '{"ok":true}',{'total_tokens':12}
        result,meta=complete(self.config(),'writer',[],call=call,sleep=lambda x:None)
        self.assertEqual(seen,['one','one','two']);self.assertEqual(meta['profile'],'b')
    def test_auth_error_does_not_switch_silently(self):
        seen=[]
        def call(p,m):seen.append(p);raise ProviderError('HTTP 401')
        with self.assertRaises(ProviderError):complete(self.config(),'writer',[],call=call,sleep=lambda x:None)
        self.assertEqual(len(seen),1)
    def test_all_profiles_exhausted_stops(self):
        def call(p,m):raise ProviderError('HTTP 503',True)
        with self.assertRaisesRegex(ProviderError,'All configured'):complete(self.config(),'writer',[],call=call,sleep=lambda x:None)

class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        for f in ['config','prompts','sources']:
            shutil.copytree(studio.ROOT/f,self.root/f)
        shutil.copytree(studio.ROOT/'content/week_01',self.root/'content/week_01')
        self.patch=patch.object(studio,'ROOT',self.root);self.patch.start()
        self.args=argparse.Namespace(command='draft',run='test',weeks=[1],artifacts=['slides.json'],config='config/providers.json',route='writer',instruction='Clarify',max_passes=2,max_jobs=2,dry_run=False,refresh=False)
        self.obj=studio.read(self.root/'content/week_01/slides.json')
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def mock_call(self,*args):return json.dumps(self.obj),{'profile':'test','usage':{}}
    def test_resume_after_model_switch_and_explicit_refresh(self):
        with patch.object(studio,'complete',side_effect=self.mock_call) as call:
            studio.run_authoring(self.args)
            cfg=self.root/'config/providers.json';data=studio.read(cfg);data['profiles']['openai_primary']['model']='different';studio.write(cfg,data)
            studio.run_authoring(self.args);self.assertEqual(call.call_count,1)
            self.args.refresh=True;studio.run_authoring(self.args);self.assertEqual(call.call_count,2)
    def test_failed_json_does_not_replace_content(self):
        before=(self.root/'content/week_01/slides.json').read_bytes()
        with patch.object(studio,'complete',return_value=('not json',{})):
            with self.assertRaises(ValueError):studio.run_authoring(self.args)
        self.assertEqual(before,(self.root/'content/week_01/slides.json').read_bytes())
        self.assertFalse((self.root/'drafts/test/.lock').exists())
    def test_apply_backup_and_stale_source_rejection(self):
        with patch.object(studio,'complete',side_effect=self.mock_call):studio.run_authoring(self.args)
        target=self.root/'content/week_01/slides.json';original=target.read_bytes()
        studio.apply(self.args);self.assertEqual(len(list((self.root/'backups').glob('*/manifest.json'))),1)
        target.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Source changed'):studio.apply(self.args)
    def test_lock_prevents_two_writers(self):
        lock=self.root/'drafts/test/.lock';lock.parent.mkdir(parents=True);lock.touch()
        with self.assertRaisesRegex(ValueError,'locked'):studio.run_authoring(self.args)
        self.assertTrue(lock.exists())
    def test_short_notes_rejected(self):
        bad=copy.deepcopy(self.obj);bad['s01']['speaker_notes']='too short'
        with self.assertRaises(ValueError):studio.validate('slides.json',bad,self.obj)

class HTTPAdapterTests(unittest.TestCase):
    def test_actual_http_transport_and_payload(self):
        import threading
        from http.server import BaseHTTPRequestHandler,HTTPServer
        from src.providers import request
        observed={}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                observed['path']=self.path
                observed['body']=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                raw=json.dumps({'choices':[{'message':{'content':'{"accepted":true}'}}],'usage':{'total_tokens':10}}).encode()
                self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(raw)
        server=HTTPServer(('127.0.0.1',0),Handler)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            text,usage=request({'base_url':f'http://127.0.0.1:{server.server_port}/v1','model':'test-model','parameters':{'max_tokens':20}},[{'role':'user','content':'Example'}])
            self.assertEqual(observed['path'],'/v1/chat/completions')
            self.assertEqual(observed['body']['model'],'test-model')
            self.assertEqual(json.loads(text),{'accepted':True});self.assertEqual(usage['total_tokens'],10)
        finally:server.shutdown();server.server_close();worker.join()
    def test_insecure_remote_endpoint_rejected(self):
        from src.providers import request
        with self.assertRaisesRegex(ProviderError,'HTTPS'):request({'base_url':'http://example.com/v1','model':'test'},[])

if __name__=='__main__':unittest.main()
