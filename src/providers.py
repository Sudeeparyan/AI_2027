"""Small configurable REST adapter. No vendor SDK or embedded credentials."""
import json, os, time, urllib.request, urllib.error
from urllib.parse import urlparse

class ProviderError(RuntimeError):
    def __init__(self, message, retryable=False, delay=0):
        super().__init__(message); self.retryable=retryable; self.delay=delay

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # Never forward a credential to a redirect target.

def request(profile, messages):
    url=profile['base_url'].rstrip('/')+'/chat/completions'
    parsed=urlparse(url)
    if parsed.scheme!='https' and not (parsed.scheme=='http' and parsed.hostname in ('localhost','127.0.0.1','::1')):
        raise ProviderError('Use HTTPS, or HTTP only for a local endpoint.')
    if parsed.username or parsed.password or parsed.query:
        raise ProviderError('Keep credentials out of endpoint URLs.')
    key=os.environ.get(profile.get('key_env',''),'')
    if profile.get('key_env') and not key:
        raise ProviderError('Missing environment variable: '+profile['key_env'])
    model=os.environ.get(profile.get('model_env',''),'') or profile.get('model','')
    if not model or model.startswith('SET_'):
        raise ProviderError('Set the model/deployment name for this profile.')
    payload={'model':model,'messages':messages,**profile.get('parameters',{})}
    headers={'Content-Type':'application/json'}
    if key: headers[profile.get('auth_header','Authorization')]=profile.get('auth_prefix','Bearer ')+key
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers=headers,method='POST')
    try:
        with urllib.request.build_opener(NoRedirect()).open(req,timeout=profile.get('timeout_seconds',120)) as response:
            obj=json.load(response)
        content=obj['choices'][0]['message']['content']
        if not isinstance(content,str): raise ValueError()
        return content,obj.get('usage',{})
    except urllib.error.HTTPError as exc:
        # Do not log response bodies: upstream errors can echo request content.
        delay=exc.headers.get('Retry-After','0')
        raise ProviderError(f'HTTP {exc.code}',exc.code in (408,429,500,502,503,504),min(float(delay),30) if delay.isdigit() else 0) from None
    except (urllib.error.URLError,TimeoutError):
        raise ProviderError('Network error or timeout',True) from None
    except (KeyError,ValueError,TypeError):
        raise ProviderError('Unsupported or malformed response') from None

def complete(config,route,messages,call=request,sleep=time.sleep):
    attempts=[]
    for name in config['routes'][route]:
        profile=config['profiles'][name]
        for attempt in range(config.get('retries_per_profile',1)+1):
            try:
                text,usage=call(profile,messages)
                return text,{'profile':name,'model':os.environ.get(profile.get('model_env',''),'') or profile.get('model'), 'usage':usage,'attempts':attempts}
            except ProviderError as exc:
                attempts.append({'profile':name,'attempt':attempt+1,'error':str(exc)})
                if not exc.retryable: raise ProviderError(json.dumps(attempts)) from None
                if attempt<config.get('retries_per_profile',1):sleep(max(exc.delay,min(2**attempt,8)))
    raise ProviderError('All configured profiles unavailable: '+json.dumps(attempts))
