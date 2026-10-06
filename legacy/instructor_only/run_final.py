from pathlib import Path
import json,sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'content/shared'))
from policy_assistant import answer
cases=json.loads((Path(__file__).with_name('final_questions.json')).read_text())
k=int(sys.argv[1]) if len(sys.argv)>1 else 1
for case in cases:
 result=answer(case['question'],case['role'],k)
 print(case['id'],'expected',case['expected_sources'],'observed',result['sources'])
