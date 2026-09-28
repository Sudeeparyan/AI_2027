import re
POLICIES = [{'id': 'P1', 'text': 'Student travel costs including bus tickets are not reimbursed.', 'role': 'student', 'version': '2027-01'}, {'id': 'P2', 'text': 'Library loans may be extended once for seven days.', 'role': 'student', 'version': '2027-01'}, {'id': 'P3', 'text': 'Students may request one deadline extension within seven days of the original due date.', 'role': 'student', 'version': '2027-01'}, {'id': 'P4', 'text': 'Academic appeals must be filed within ten days of the result.', 'role': 'student', 'version': '2027-01'}, {'id': 'P5', 'text': 'Staff payroll changes require manager approval before any record is changed.', 'role': 'staff', 'version': '2027-01'}]
STOP={'a','an','the','is','are','to','of','my','can','i','what','how','for','in','and'}
def tokens(text):return set(re.findall(r'[a-z]+',text.lower()))-STOP
def rank(question,role='student',k=1,corpus=None):
    if not isinstance(question,str) or not question.strip():raise ValueError('Question must be nonempty text')
    if role not in ('student','staff'):raise ValueError('Unknown trusted role')
    if k not in (1,2):raise ValueError('k must be 1 or 2')
    corpus=POLICIES if corpus is None else corpus
    allowed=[p for p in corpus if p['role']=='student' or role=='staff']
    scored=sorted([(len(tokens(question)&tokens(p['text'])),p) for p in allowed],key=lambda x:x[0],reverse=True)
    return [p for score,p in scored if score>=2][:k]
def answer(question,role='student',k=1,corpus=None):
    found=rank(question,role,k,corpus)
    return {'status':'extract' if found else 'unsupported','text':' '.join(p['text'] for p in found),
            'sources':[p['id'] for p in found],'version':'2027-01'}
def safe_trace(request_id,result):
    return {'request_id':request_id,'status':result['status'],'sources':result['sources'],'version':result['version']}
