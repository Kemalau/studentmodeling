"""Read a C4 AML log without exposing temporary signed content URLs."""
import argparse
import json
import subprocess
import urllib.parse
import urllib.request

JOBS={'A':'stoic_queen_l2zrtkcpcq','B':'tidy_card_gdqp59xzfl'}
BASE=('https://ml.azure.com/api/eastus/artifact/v2.0/subscriptions/'
    'b6dc87f3-c479-49c8-8cb5-7896da3ff895/resourceGroups/'
    'rg-cs-newsandfeeds-singularity/providers/Microsoft.MachineLearningServices/'
    'workspaces/CS-NewsAndFeeds-Singularity/artifacts/ExperimentRun/')

def read(arm,lines=50,path='user_logs/std_log_process_0.txt'):
    query=urllib.parse.urlencode({'path':path})
    result=subprocess.run(['az','rest','--method','get','--resource','https://management.azure.com/',
        '--url',BASE+'dcid.'+JOBS[arm]+'/contentinfo?'+query,'-o','json'],
        text=True,capture_output=True,timeout=45)
    if result.returncode:raise RuntimeError('Azure contentinfo failed: '+result.stderr[:1000])
    uri=json.loads(result.stdout)['contentUri']
    try:
        with urllib.request.urlopen(uri,timeout=45) as stream:
            content=stream.read().decode('utf-8',errors='replace')
    except Exception as exc:
        # Do not include an exception URL containing a signed credential.
        raise RuntimeError('Log content fetch failed: '+type(exc).__name__) from None
    return '\n'.join(content.splitlines()[-lines:])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('arm',choices=JOBS)
    p.add_argument('--lines',type=int,default=50)
    p.add_argument('--path',default='user_logs/std_log_process_0.txt');a=p.parse_args()
    print(read(a.arm,a.lines,a.path))
