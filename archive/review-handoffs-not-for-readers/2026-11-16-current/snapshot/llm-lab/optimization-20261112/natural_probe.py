#!/usr/bin/env python3
"""Small, fixed, nonrepetitive functional tasks; synthetic data only."""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,time,urllib.request
R=Path(__file__).resolve().parent;LAB=R.parent;CONFIG=LAB/'deepseek41-bringup-20261101/server-config.json'
spec=importlib.util.spec_from_file_location('helpers',LAB/'postcompact-20261103/live_probe.py');helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
TASKS=[
('ledger', '''Calculate the closing balance of this fictional cash ledger. The opening balance is 1000 units. Apply the following entries in the order shown:
09:10 sale received: +137
09:35 supplies paid: -48
10:20 refund received: +92
11:05 delivery paid: -31
12:15 materials paid: -76
14:40 invoice received: +205
16:30 maintenance paid: -19
All entries are final. There are no taxes, fees, pending transactions, currency conversions, or hidden adjustments. Include every listed entry exactly once. The opening balance is the starting amount and must not be added again as an extra transaction. Return only a JSON object with one integer field named "balance". Do not include intermediate calculations or commentary.''', {'balance':1260}),
('filter', '''Process the integer list below using these exact rules. First remove repeated values, keeping each distinct integer only once. Then keep only the integers strictly less than zero. Finally sort the remaining integers in ascending numerical order, from the smallest value to the largest.
Input: [12, -7, 3, 12, 0, -19, 8, 3, 5, -7, 21, -3, 2, 0, 17, -19]
Zero is not negative. Positive values must not appear in the result. Preserve the minus signs; do not sort by the printed characters or by absolute value. Return only the resulting JSON array of integers. Do not wrap the array in an object, and do not include explanations, markdown, or a description of the steps.''', [-19,-7,-3]),
('warehouse', '''Use this fictional warehouse table. Each row is a separate product. Quantity means number of units, and unit_price is the price of one unit in whole credits.
product | quantity | unit_price
P01 | 4 | 11
P02 | 8 | 6
P03 | 5 | 9
P04 | 3 | 25
P05 | 9 | 4
P06 | 2 | 31
P07 | 7 | 12
P08 | 6 | 15
Find product P07 and multiply its quantity by its unit_price. Ignore the other products when calculating this total. No discounts, fees, taxes, or rounding apply. Return only a JSON object with the string field "id" set to the product identifier and the integer field "total" set to the calculated amount. Do not add any other fields or explanatory prose.''', {'id':'P07','total':84}),
('japanese-stock', '''次の架空の在庫記録だけを使って、閉店時の在庫数を計算してください。対象は同じ商品の個数です。
開店時の在庫は24個でした。午前中に7個が入荷し、その後5個を発送しました。午後には返品された2個が検品を通過して、販売可能な在庫へ戻りました。夕方にさらに9個を発送して閉店しました。
返品の2個は在庫に加算してください。記載していない入出庫、破損、予約、別倉庫への移動はありません。開店時の24個を二重に数えないでください。
回答はキーがstockだけのJSONオブジェクトにしてください。値は半角数字の整数です。計算過程、単位、説明、マークダウンは付けないでください。''', {'stock':19}),
]
def main():
    p=argparse.ArgumentParser();p.add_argument('--label',required=True);a=p.parse_args();assert a.label.replace('-','').replace('_','').isalnum()
    lock=(R/'.benchmark.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    raw=CONFIG.read_bytes();c=json.loads(raw);key=Path(c['api_key_file']).read_text().strip();headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'}
    base=f"http://{c['host']}:{c['port']}";opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def query(path,body=None):
        with opener.open(urllib.request.Request(base+path,headers=headers,data=None if body is None else json.dumps(body,ensure_ascii=False).encode()),timeout=240) as res:return json.load(res)
    assert query('/health').get('status')=='ok'
    if any(x.get('is_processing') for x in query('/slots')):raise RuntimeError('API busy')
    inputs=R/'NATURAL_INPUTS.json'
    if not inputs.exists():
        cases=[]
        for name,text,expected in TASKS:
            formatted=query('/apply-template',{'messages':[{'role':'user','content':text}]})['prompt']
            ids=query('/tokenize',{'content':formatted,'add_special':True,'parse_special':True})['tokens']
            assert 128<=len(ids)<512,(name,len(ids))
            cases.append({'name':name,'content':text,'formatted':formatted,'tokens':ids,'expected':expected})
        inputs.write_text(json.dumps({'cases':cases},indent=2,ensure_ascii=False)+'\n')
    fixed=json.loads(inputs.read_text());folder=R/'natural'/f'{time.strftime("%Y%m%d-%H%M%S")}-{a.label}';folder.mkdir(parents=True)
    record={'label':a.label,'config_sha256':hashlib.sha256(raw).hexdigest(),'inputs_sha256':hashlib.sha256(inputs.read_bytes()).hexdigest(),'cases':[],'complete':False}
    def save():
        p=folder/'RESULT.tmp';p.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n');p.replace(folder/'RESULT.json')
    save()
    for case in fixed['cases']:
        helpers.watch();assert CONFIG.read_bytes()==raw
        body={'prompt':case['tokens'],'n_predict':96,'temperature':0,'seed':42,'cache_prompt':False,'stream':True,'return_tokens':True}
        start=time.monotonic();first=None;out=[];final=None
        with opener.open(urllib.request.Request(base+'/completion',data=json.dumps(body).encode(),headers=headers),timeout=240) as res:
            for rawline in res:
                line=rawline.decode().strip()
                if not line.startswith('data:'):continue
                payload=line[5:].strip()
                if payload=='[DONE]':break
                event=json.loads(payload)
                if event.get('error'):raise RuntimeError('API error')
                content=event.get('content','');out.append(content)
                if first is None and (content or event.get('tokens')):first=time.monotonic()
                if event.get('stop'):final=event;break
        assert final is not None
        content=''.join(out).strip();parsed=None
        try:parsed=json.loads(content)
        except json.JSONDecodeError:pass
        row={'name':case['name'],'prompt_tokens':len(case['tokens']),'content':content,'correct':parsed==case['expected'],
             'ttft_s':first-start if first else None,'elapsed_s':time.monotonic()-start,'timings':final['timings'],
             'stop_type':final.get('stop_type'),'output_sha256':hashlib.sha256(content.encode()).hexdigest()}
        assert row['timings']['cache_n']==0 and row['timings']['prompt_n']==len(case['tokens'])
        record['cases'].append(row);save();helpers.watch();print(json.dumps({'label':a.label,**row},ensure_ascii=False),flush=True)
    record['complete']=True;record['all_correct']=all(x['correct'] for x in record['cases']);save()
if __name__=='__main__':main()
