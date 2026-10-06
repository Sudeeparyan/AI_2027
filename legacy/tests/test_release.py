import json, shutil, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import course, studio

class ReleaseTests(unittest.TestCase):
    def test_failed_checks_preserve_current_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for rel in ['content/week_01','src']:
                shutil.copytree(course.ROOT/rel,root/rel)
            shutil.copy2(course.ROOT/'content/references.json',root/'content/references.json')
            shutil.copytree(course.ROOT/'content/teaching_design',root/'content/teaching_design')
            shutil.copy2(course.ROOT/'course.py',root/'course.py')
            current=root/'outputs/current/week_01';current.mkdir(parents=True)
            sentinel=current/'release.txt';sentinel.write_text('approved release')
            with patch.object(course,'ROOT',root),patch.object(course,'audit',return_value=[{'kind':'notebook','message':'intentional test failure'}]),patch.object(course,'export_visuals') as export:
                with self.assertRaisesRegex(RuntimeError,'not replaced'):course.build([1],1,True)
                export.assert_not_called()
            self.assertEqual(sentinel.read_text(),'approved release')
            report=json.loads(next((root/'outputs/runs').glob('*/report.json')).read_text())
            self.assertEqual(report['weeks']['week_01']['remaining_issues'],1)
    def test_candidate_cannot_redirect_shared_code(self):
        original={'nbformat':4,'cells':[{'cell_type':'code','source':'print(1)','metadata':{'source_file':'content/shared/policy_assistant.py'}}]}
        changed=json.loads(json.dumps(original));changed['cells'][0]['metadata']['source_file']='../../private.py'
        with self.assertRaisesRegex(ValueError,'Preserve shared'):studio.validate('lab.ipynb',changed,original)
if __name__=='__main__':unittest.main()
