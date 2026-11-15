"""Fill numeric LaTeX slots only from a passed independently audited summary."""
import argparse
import json
import math
from pathlib import Path
import re
from audit_scores import Reader,sha
from independent_core import require
from summary_tables import key


def render(audit,summary,registry,summary_binding,slots_binding):
    require(audit.get('schema')=='pare.formal-score-audit.v1' and audit.get('status')=='audit_passed'
            and audit.get('errors')==[] and audit.get('jobs')==315,'No complete independent formal-score audit')
    require(audit['summary']==summary_binding and audit['result_slots']==slots_binding,'Audit/summary/slots binding differs')
    require(summary.get('schema')=='pare.formal-score-summary.v1'
            and summary.get('status')=='derived_from_independent_score_audit'
            and summary.get('result_slots_sha256')==slots_binding['sha256'],'Unaudited summary')
    values={key(r):r for r in summary['measurements']}
    require(len(values)==len(summary['measurements']),'Duplicate measurement identities')
    require(len({s['slot_id'] for s in registry['slots']})==len(registry['slots']),'Duplicate manuscript slot ID')
    lines=['% Numeric results from a passed independent formal-score audit only.',
           '% Manual text slots are intentionally left unchanged.']
    bound=[];manual=[]
    for slot in registry['slots']:
        if slot['kind']=='manual_text':manual.append(slot['slot_id']);continue
        require(slot['kind']=='measurement','Unknown slot kind')
        require(slot.get('value') is None,'Registry must not prefill an unaudited number')
        require(key(slot) in values,'Missing exact audited measurement: '+slot['slot_id'])
        value=values[key(slot)]['value']
        require(type(value) in (int,float) and math.isfinite(value),'Null/nonfinite audited number: '+slot['slot_id'])
        fmt=slot.get('format','.2f')
        require(re.fullmatch(r'\+?\.[0-9]f',fmt),'Unsupported numeric format')
        require(re.fullmatch('[a-zA-Z0-9_./-]+',slot['slot_id']),'Unsafe TeX slot ID')
        formatted=format(value,fmt)
        lines.append('\\expandafter\\def\\csname result@'+slot['slot_id']+'\\endcsname{'+formatted+'}')
        bound.append({'slot_id':slot['slot_id'],'measurement_key':list(key(slot)),'value':value,'formatted':formatted})
    return '\n'.join(lines)+'\n',bound,manual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','summary','slots'):
        parser.add_argument('--'+name,required=True,type=Path);parser.add_argument('--'+name+'-sha256',required=True)
    parser.add_argument('--out',required=True,type=Path);args=parser.parse_args();reader=Reader()
    bindings={name:{'path':str(getattr(args,name).resolve()),'sha256':getattr(args,name+'_sha256')}
              for name in ('audit','summary','slots')}
    data={name:reader.json(binding) for name,binding in bindings.items()}
    text,bound,manual=render(data['audit'],data['summary'],data['slots'],bindings['summary'],bindings['slots'])
    receipt=args.out.with_suffix('.binding.json')
    reader.finish();require(not args.out.exists() and not receipt.exists(),'Refuse to overwrite result macro file or receipt')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as stream:stream.write(text)
    with receipt.open('x') as stream:json.dump({'status':'bound_from_passed_formal_audit','sources':bindings,
        'tex_sha256':sha(args.out),'bound_measurements':bound,'manual_slots_not_filled':manual},stream,indent=2)
    print(json.dumps({'status':'bound','numeric_slots':len(bound),'manual_slots':len(manual),'out':str(args.out)}))


if __name__=='__main__':main()
