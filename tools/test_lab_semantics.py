"""Scientific regression fixtures; no model, network, keys or server processes."""
import ast, json, re, time, math, operator, hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]

def definitions(week):
    text=(ROOT/f'curriculum/labs/week_{week:02d}_lab.py').read_text(encoding='utf-8')
    tree=ast.parse('\n'.join('' if line.startswith('%pip ') else line for line in text.splitlines()))
    return {node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}

def load(defs,names,scope):
    for name in names:
        exec(compile(ast.Module(body=[defs[name]],type_ignores=[]),'<source fixture>','exec'),scope)

report={}
d=definitions(12)
s={'ast':ast,'json':json,'re':re,'time':time,'math':math,'operator':operator,'check':lambda good,*args:bool(good)}
load(d,['chunk','check_chunk','citation_support','check_citation_support','task_evidence','resource_metrics'],s)
report['chunk_exact_windows']=s['check_chunk']()
original=s['chunk']
s['chunk']=lambda text,size=220,overlap=50:[text]*3 if text.strip() else []
report['triplicated_whole_text_rejected']=not s['check_chunk']()
s['chunk']=original
report['citation_support_fixtures']=s['check_citation_support']()
task='I scored 72 in the group project and 64 in the exam. Using the weights in the handbook, what is my final mark?'
search={'call':{'name':'search_handbook'},'observation':'Assessment consists of a group project worth 60% of the final mark and an individual written exam worth 40%.'}
def calc(expression,observation='68.8'):
    return {'call':{'name':'calculator','arguments':{'expression':expression}},'observation':observation}
grade=lambda trace,answer='Final mark: 68.8':s['task_evidence'](task,answer,trace)
report.update(valid_derived_grade=grade([search,calc('60/100*72+40/100*64')]),
              equivalent_parenthesised_grade=grade([search,calc('(60*72+40*64)/100')]),
              decimal_coefficients_grade=grade([search,calc('.6*72+.4*64')]),
              constant_shortcut_rejected=not grade([search,calc('68.8')]),
              zero_times_marks_rejected=not grade([search,calc('68.8+0*72+0*64')]),
              missing_search_rejected=not grade([calc('.6*72+.4*64')]),
              late_search_rejected=not grade([calc('.6*72+.4*64'),search]),
              wrong_observation_rejected=not grade([search,calc('.6*72+.4*64','70')]),
              wrong_answer_rejected=not grade([search,calc('.6*72+.4*64')],'Final mark: 70'),
              reversed_weights_rejected=not grade([search,calc('.4*72+.6*64')]),
              zero_success_ratio=s['resource_metrics']([{'success':False,'seconds':1,'model_calls':0,'tool_calls':1}])['seconds_per_success'] is None)
report['percentage_grade']=grade([search,calc('60%*72+40%*64')])
report['unary_minus_grade']=grade([search,calc('.6*72 - -.4*64')])
for label, project, exam, result in [('duplicate_marks',72,72,72), ('mark_weight_collision',60,40,52)]:
    prompt=f'I scored {project} in the group project and {exam} in the exam. Using the weights in the handbook, what is my final mark?'
    trace=[search,calc(f'60/100*{project}+40/100*{exam}',str(result))]
    report[label]=s['task_evidence'](prompt,f'Final mark: {result}',trace)
assert all(report.values()),report

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from typing import Literal
d=definitions(11)
s={'time':time,'check':lambda good,*args:bool(good),'FastAPI':FastAPI,'BaseModel':BaseModel,'Field':Field,'Literal':Literal,'MODEL_VERSION':'fixture','CORE_TOKENS':24,'stream_chat':None,'tok':None}
s['api']=FastAPI()
load(d,['GeneratedText','termination_reason','measure','check_measure','ChatMessage','ChatRequest','chat_completions','check_endpoint'],s)
report['cap_reports_length']=s['termination_reason']([3,4],2,[99])=='length'
report['eos_at_cap_reports_stop']=s['termination_reason']([3,99],2,[99])=='stop'
report['early_eos_reports_stop']=s['termination_reason']([99],2,99)=='stop'
report['buffered_chunk_rate']=s['check_measure'](s['measure'])
s['generate']=lambda messages,max_new_tokens=128:('Real fixture answer',7,2)
with TestClient(s['api']) as client:
    response=client.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'Hello'}]})
    payload=response.json()
    s.update(test_client=client,api_response=response,choices=payload['choices'],api_usage=payload['usage'],api_answer=payload['choices'][0]['message']['content'])
    report['endpoint_forwarding']=s['check_endpoint']()
    saved=s['generate']
    s['check_endpoint']()
    report['endpoint_restores_generator']=s['generate'] is saved
original=s['chat_completions']
exec('def broken(req: ChatRequest):\n    return {"choices":[{"message":{"content":""}}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}',s)
app=FastAPI();app.post('/v1/chat/completions')(s['broken'])
s['chat_completions']=s['broken']
with TestClient(app) as client:
    response=client.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'Hello'}]})
    payload=response.json()
    s.update(test_client=client,api_response=response,choices=payload['choices'],api_usage=payload['usage'],api_answer='')
    report['fabricated_endpoint_rejected']=not s['check_endpoint']()
assert all(report.values()),report

d=definitions(10)
s={'re':re,'check':lambda good,*args:bool(good)}
load(d,['score','check_score','yes_no_label','check_yes_no_label'],s)
report['signed_numeric_scoring']=s['check_score']()
report['unrelated_yes_no_prefixes_rejected']=s['check_yes_no_label']()
assert all(report.values()),report
(ROOT/'build/lab_semantics_check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
